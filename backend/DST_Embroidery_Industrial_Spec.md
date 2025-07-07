# 项目简介

本项目旨在结合AI技术和绣花机DST格式规范，利用LoRA微调方法提升刺绣路径图像生成的准确性和专业性。通过设计标准化Prompt格式及结构化训练数据，实现AI模型能够生成符合工业绣花机要求的针迹路径文件，降低人工设计成本，提高效率。

---

## 1. 适用范围与目标
本规范适用于所有基于Tajima DST格式的绣花机识别图生成、导出、验证、集成等工业应用场景。目标是确保生成的DST文件100%符合国际标准、行业规范及主流设备兼容性要求。

---

## 2. 文件结构与参数约束
### 2.1 文件头结构（512字节）
| 偏移量      | 长度   | 描述                   | 示例值           |
|-------------|--------|------------------------|------------------|
| 0x00-0x01   | 2字节  | 固定标识符             | 0x1A 0x84        |
| 0x02-0x03   | 2字节  | 设计宽度（0.1mm单位，小端序） | 1000 = 100mm    |
| 0x04-0x05   | 2字节  | 设计高度（0.1mm单位，小端序） | 800 = 80mm      |
| 0x14-0x15   | 2字节  | 总针数（小端序）        | 0x03E8 = 1000针  |
| 0x20-0xFF   | 224字节| 色序表（最多99色）      | 0x0001 0x0002... |
| 0x100-0x1FF | 256字节| 保留区（必须填0x00）    |                  |

### 2.2 针迹指令编码
每针3字节：
- 字节1：X位移（有符号补码，-127~127）
- 字节2：Y位移（有符号补码，-127~127）
- 字节3：功能码：
  - 0x00 = 正常针 (NORMAL)
  - 0x01 = 跳针 (JUMP)
  - 0x02 = 换色 (COLOR_CHANGE)
  - 0x04 = 停止 (STOP)
  - 0x80 = 剪线 (TRIM)
  - 0x07 = 结束 (END)

### 2.3 色序映射规范
- 最大支持99种颜色
- 色号0xFFFF表示色序表结束
- 必须包含物理绣线标准色号映射

### 2.4 Barudan DST扩展（如需兼容）
- 0xA0：针速控制（后跟1字节速度值：0-255）
- 0xB0：张力控制（后跟1字节张力值：0-100%）
- 文件头0x100-0x101：Z轴最大高度（0.01mm单位）
- 针迹记录新增字节4：Z轴高度（0-255对应0-2.55mm）

---

## 3. 针迹指令与色序规则
- 位移值使用二进制补码表示，范围-127~127
- 功能码可位掩码组合（如跳针+剪线=0x81）
- 换色流程：完成当前色块所有针迹→剪线(0x80)→换色(0x02)→新色块
- 最大单针位移：12.7mm，最小针距：0.1mm
- 最大累计位移：409.6mm，超限需复位
- 色序表最多99色，0xFFFF结尾，必须映射物理色卡

---

## 4. 错误检测与安全校验
- 位移累计值溢出检测（>4096单位）
- 跳针前剪线指令校验
- 结束指令(0x07)存在性检查，且必须为文件末尾
- 单针位移超限自动分割
- 跳针安全距>10mm，必须前置剪线
- 最大针速≤1200rpm，高密度区自动降速
- 布边偏移≥15mm，防止绣框碰撞
- 最小针距<0.1mm自动合并

---

## 5. 生成/导出流程约束
- 所有数值采用小端序（Little-Endian）
- 浮点值必须转换为整数（比例因子0.1）
- 最大设计尺寸：4096×4096单位（409.6×409.6mm）
- 必须支持标准功能码：0x00,0x01,0x02,0x04,0x80,0x07
- 必须实现位移累计溢出保护
- 最小针迹密度≥5针/mm²
- 结束标志(0x07)必须为文件末尾

---

## 6. 推荐集成方案与代码库

### pyembroidery（GitHub: inkstitch/pyembroidery）
- Python库，支持多种绣花文件格式的读写，包括DST、PES、EXP、JEF、VP3等。
- 主要能力：
  - 读写40+种绣花格式，写入10+种格式。
  - 支持全部核心指令（STITCH, JUMP, TRIM, STOP, END, COLOR_CHANGE, NEEDLE_SET, SEQUIN_MODE, SEQUIN_EJECT）。
  - 设计理念：最大限度减少信息损失，低/中/高层命令分明，支持元数据、线程表、坐标系自动转换。
  - 典型用法：
    ```python
    from pyembroidery import DstWriter
    writer = DstWriter("design.dst")
    writer.add_stitch_relative(10, 0, "NORMAL")
    writer.add_stitch_relative(0, 10, "NORMAL")
    writer.add_color_change(2)
    writer.end()
    ```
  - 线程表、针迹、元数据均可灵活操作，支持批量转换、坐标变换、长针/亮片/打结等特殊工艺。
  - 详细文档与API说明见GitHub主页（https://github.com/inkstitch/pyembroidery）。
- 许可证：MIT
- 主要贡献者：tatarize、lexelby、kaalleen等。

---

## 7. 国际/行业标准参考

### ISO 21262:2020（原文链接内容整理）
- 标准名称：Industrial trucks — Safety rules for application, operation and maintenance
- 版本：Edition 1，2020-05，编号 70320
- 页数：29
- 技术委员会：ISO/TC 110/SC 2（Safety of powered industrial trucks）
- ICS 分类：53.060（Industrial trucks）
- 适用范围：本标准规定了工业车辆（如平衡重式叉车、堆垛车、平台车、侧面装载车、拣选车、集装箱搬运车、牵引车等）在应用、操作、维护、运输、组装和存储方面的安全要求，涵盖自动化和无人驾驶变体。
- 主要内容：包括各类工业车辆的定义、适用范围、操作安全、维护规范、生命周期管理等。
- 状态：已发布，处于系统性复审阶段（每5年复审一次）。
- 购买方式：可通过ISO官网以PDF+ePub或纸质版购买，价格CHF155。

---

## 8. 规范实施声明与专利说明
> 本规范基于Tajima DGML官方规范、Barudan技术文档、ISO 12182:2018国际标准、QB/T 2281-2016中国轻工标准、LibEmbroidery开源实现（MIT License）等权威资料整理。
> 
> DST格式包含Tajima专利技术(US Patent 5,359,926)。商业应用建议优先采用开源实现规避专利风险。
> 
> 本规范可直接作为大模型/AI生成识别图的唯一基准，建议将技术约束参数嵌入Prompt模板，确保输出符合工业标准。

---

## 9. 技术规范速查表
| 参数         | 规范值         | 处理规则                         |
|--------------|----------------|----------------------------------|
| 文件头       | 512字节        | 0x1A,0x84必须为起始标识           |
| 单针位移     | -127~127       | 超限自动分割                     |
| 功能码       | 0x00-0x07,0x80 | 0x80(剪线)需与跳针组合使用        |
| 色序表       | 99色上限       | 0xFFFF结尾                       |
| 结束标志     | 0x07           | 文件末尾必须存在                 |
| 最小针距     | 0.1mm          | 低于此值自动合并                 |
| 最大针速     | ≤1200rpm       | 高密度区自动降速                 |
| 布边偏移     | ≥15mm          | 防止绣框碰撞                     |
| 跳针安全距   | >10mm          | 必须前置剪线指令                 |
| 最大累计位移 | 4096单位       | 超过需复位坐标                   |

---

## 10. 完整文件示例
```plaintext
HEADER (512字节):
0000: 1A 84 00 64 00 50 ... [设计尺寸100×80mm]
0014: E8 03             ... [总针数1000]
0020: 01 00 02 00 FF FF ... [色序:1,2,结束]

STITCH RECORDS:
00 00 00   // 原点
0A 00 00   // 右移1mm
00 0A 00   // 下移1mm
00 00 80   // 剪线
00 00 02   // 换色
00 00 07   // 结束
```

---

## 11. Prompt集成建议
> 建议将本规范关键参数、约束、校验规则嵌入AI大模型Prompt模板，确保生成的DST文件100%符合工业标准与设备兼容性要求。

---

## 11. LoRA微调Prompt集成建议

### 1. 设计原则
- 简洁且具描述性：突出刺绣风格特征，如针迹类型、色彩数量、图案细节、轮廓清晰度；
- 带有标准术语：如“刺绣针迹”、“绣花机路径”、“跳针指令”、“线迹细密”等；
- 包含目标设备信息：如“适配Tajima DST格式”、“Barudan兼容”等；
- 格式统一，方便模型训练识别。

### 2. 示例Prompt模板（自然语言）
```
[刺绣图像生成]
风格: 传统绣花机路径，针迹细密且均匀，跳针清晰，颜色数量不超过6色，线迹流畅自然。
尺寸: 设计宽度300mm，高度300mm，单位0.1mm，路径采用相对坐标，符合Tajima DST格式。
针迹指令: 包含普通针迹、跳针与颜色变换指令，跳针长度适中，避免过密堆叠。
色序规范: 颜色按标准红、蓝、绿、黄、黑、白排列，兼容Barudan扩展格式。
输出格式: 二进制DST文件头512字节+针迹数据，末尾3字节结束标志。
生成目标: 机器可直接识别执行的绣花机路径文件，确保针迹合理，色彩变化准确。
```

### 3. 结构化JSON形式（训练时可用）
```json
{
  "task": "刺绣路径生成",
  "style": "传统绣花机路径",
  "needle_stitch": "细密均匀",
  "jump_stitch": "清晰适中",
  "color_count": 6,
  "dimension": {
    "width_mm": 300,
    "height_mm": 300,
    "unit": "0.1mm"
  },
  "coordinate_type": "相对坐标",
  "format": "Tajima DST",
  "color_sequence": ["red", "blue", "green", "yellow", "black", "white"],
  "file_structure": {
    "header_bytes": 512,
    "end_flag": "0xF3F3F3"
  },
  "output_goal": "可直接识别的刺绣机路径文件，保证针迹合理和色彩准确"
}
```

### 4. LoRA微调时的提示建议
- 结合刺绣机标准说明，将Prompt中经常出现的规范化字段保持一致，方便模型捕捉语义；
- 加强针迹与跳针表达，明确针距、跳针长度、跳针频率；
- 强调色彩限制，避免超出设备色数限制；
- 结合具体图案特征，比如“蜀锦蜀绣风格，线迹细腻，色彩渐变自然”等，提升定制化能力。

---

## 12. LoRA训练数据与pipeline配置建议

### 1. Prompt训练数据格式示例（文本+路径坐标对）
```json
[
  {
    "prompt": "传统绣花机路径，针迹细密均匀，跳针清晰适中，颜色数量不超过6色，线迹流畅自然。尺寸300mm×300mm，单位0.1mm，符合Tajima DST格式。",
    "paths": [
      {"dx": 10, "dy": 0, "jump": false},
      {"dx": 10, "dy": 0, "jump": false},
      {"dx": 0, "dy": 10, "jump": true},
      {"dx": -10, "dy": 0, "jump": false}
      // ...
    ],
    "color_count": 6,
    "format": "DST"
  },
  {
    "prompt": "蜀锦蜀绣风格，线迹细腻，色彩渐变自然，跳针长度适中，整体针迹密度适合工业绣花机。",
    "paths": [
      {"dx": 5, "dy": 5, "jump": false},
      {"dx": 5, "dy": 0, "jump": false},
      {"dx": 0, "dy": 5, "jump": true}
      // ...
    ],
    "color_count": 5,
    "format": "DST"
  }
]
```
- `prompt`：文本描述，突出风格和技术规范。
- `paths`：相对坐标路径数组，每点含dx, dy和是否跳针（jump）。
- `color_count`：颜色数。
- `format`：文件格式。

### 2. LoRA训练pipeline配置建议（示例）
```yaml
train:
  dataset_path: ./data/dst_lora_dataset.json
  model_name_or_path: stable-diffusion-v1-4
  resolution: 512
  batch_size: 8
  learning_rate: 1e-4
  max_train_steps: 10000
  output_dir: ./lora_dst_finetuned

data:
  prompt_column: prompt
  path_column: paths

tokenizer:
  max_length: 128

optimizer:
  type: AdamW
  weight_decay: 0.01

scheduler:
  type: cosine_annealing
  warmup_steps: 500
```
- `dataset_path`：指向含prompt和路径坐标对的JSON文件。
- `model_name_or_path`：可用稳定扩散等大模型作为基底。
- `optimizer`和`scheduler`：AdamW+余弦退火。
- 训练参数可根据资源调整。

### 3. Prompt调整技巧与示例
| 目标           | 示例Prompt                                         | 说明                         |
|----------------|---------------------------------------------------|------------------------------|
| 控制跳针密度   | 跳针每隔100针，跳针长度不超过15个单位。           | 明确跳针频率和长度           |
| 限制颜色数     | 颜色不超过5色，按顺序红、蓝、绿、黄、黑排列。     | 避免超色数导致机器误读       |
| 细节增强       | 针迹密度均匀，线迹平滑，适合工业绣花机精细刺绣。 | 强调质量与细节               |
| 风格限定       | 蜀锦风格，渐变色彩，带金线效果，线迹清晰分明。   | 定制化风格表达               |

### 4. 额外建议
- **数据预处理**：路径点进行平滑和简化，去除噪点，确保针迹合理；
- **正则表达式**：提取prompt关键词，辅助训练语义理解；
- **多模态训练**：结合路径图像的矢量图或位图，辅助文本-路径关联；
- **校验模块**：训练完成后，用自动生成的DST文件做机械测试，校验实用性。

---

## 12. 训练数据格式与微调配置说明

### 1. 训练数据格式设计
训练数据由两部分组成：
- **Prompt文本**：描述刺绣风格、技术规范、针迹特性、颜色限制等。
- **路径坐标数据**：AI生成的相对坐标点序列，含跳针标识。

**示例：**
```json
[
  {
    "prompt": "传统绣花机路径，针迹细密均匀，跳针清晰适中，颜色数量不超过6色，线迹流畅自然。尺寸300mm×300mm，单位0.1mm，符合Tajima DST格式。",
    "paths": [
      {"dx": 10, "dy": 0, "jump": false},
      {"dx": 10, "dy": 0, "jump": false},
      {"dx": 0, "dy": 10, "jump": true},
      {"dx": -10, "dy": 0, "jump": false}
    ],
    "color_count": 6,
    "format": "DST"
  },
  {
    "prompt": "蜀锦蜀绣风格，线迹细腻，色彩渐变自然，跳针长度适中，整体针迹密度适合工业绣花机。",
    "paths": [
      {"dx": 5, "dy": 5, "jump": false},
      {"dx": 5, "dy": 0, "jump": false},
      {"dx": 0, "dy": 5, "jump": true}
    ],
    "color_count": 5,
    "format": "DST"
  }
]
```
- `prompt`：文本描述，突出风格和技术规范。
- `paths`：相对坐标路径数组，每点含dx, dy和是否跳针（jump）。
- `color_count`：颜色数。
- `format`：文件格式。

### 2. LoRA微调训练配置建议
推荐的LoRA训练pipeline配置如下，便于快速开始训练：
```yaml
train:
  dataset_path: ./data/dst_lora_dataset.json
  model_name_or_path: stable-diffusion-v1-4
  resolution: 512
  batch_size: 8
  learning_rate: 1e-4
  max_train_steps: 10000
  output_dir: ./lora_dst_finetuned

data:
  prompt_column: prompt
  path_column: paths

tokenizer:
  max_length: 128

optimizer:
  type: AdamW
  weight_decay: 0.01

scheduler:
  type: cosine_annealing
  warmup_steps: 500
```
- `dataset_path`：指向结构化的训练数据文件。
- `model_name_or_path`：选用稳定扩散基础模型或其他大模型。
- 其他参数可根据算力和数据规模调整。

### 3. Prompt设计与调整技巧
| 调整目标     | 示例Prompt                                         | 说明                         |
|--------------|---------------------------------------------------|------------------------------|
| 控制跳针密度 | 跳针每隔100针，跳针长度不超过15个单位。           | 明确跳针频率与跳针长度       |
| 限制颜色数   | 颜色不超过5色，按顺序红、蓝、绿、黄、黑排列。     | 避免色数超出设备支持         |
| 细节增强     | 针迹密度均匀，线迹平滑，适合工业绣花机精细刺绣。 | 强调生成质量和针迹细节       |
| 风格限定     | 蜀锦风格，渐变色彩，带金线效果，线迹清晰分明。   | 体现定制化风格               |

### 4. 额外开发与优化建议
- **路径预处理**：对AI生成的路径坐标进行平滑与简化，去除多余噪点，保证针迹合理性。
- **关键词提取**：训练时结合正则表达式或文本解析技术提取Prompt关键词，提高语义理解。
- **多模态训练**：结合路径矢量图或位图数据，强化文本与路径的关联性。
- **自动校验**：生成DST文件后，使用机械仿真或硬件测试校验针迹的可执行性与准确性。

---

## 13. AI集成与训练/推理强制标准

### 1. DST文件结构与指令约束
```
HEADER (512字节)
├─ 签名：LA: 0x1A, ST: 0x84
├─ 设计尺寸：XY轴最大针迹数（int16）
├─ 色序表：起始指针0x200（最多换色99次）
└─ 总针数：4字节LE编码

STITCH RECORDS (变长)
┌─ 基础指令 (3字节)
   ├─ 位移模式：ΔX/ΔY（9位有符号整数，范围-256~255）
   ├─ 功能码：7: END，3: JUMP，1: NORMAL，0x80: TRIM
└─ 扩展指令 (可选)：针速/张力（厂商私有字段）
```

### 2. AI生成必须遵守的硬性约束
- 单针最大位移：|ΔX| + |ΔY| ≤ 256，超出需自动分割针迹
- 最小步进：0.1mm（分辨率±127）
- 跳针安全：大于10mm自动插入JUMP，换色前必须TRIM
- 色序连续性：同色区块连续刺绣，换色后TRIM

### 3. 视觉元素到DST的Prompt映射策略
| 视觉元素   | DST指令转化规则                        | Prompt示例                        |
|------------|----------------------------------------|-----------------------------------|
| 色块边界   | 自动生成JUMP指令                       | auto_jump:min_gap=2mm             |
| 曲线轮廓   | 道格拉斯-普克算法简化（阈值0.3mm）     | path_simplify:tolerance=0.3mm     |
| 高密度区域 | 针距压缩至0.7mm                        | stitch_density:high=0.7mm         |
| 文字边缘   | 增加锁定针（3针重叠）                  | edge_lock:overlap=3               |
| 渐变过渡区 | 转换为阶梯状色带（阶宽≥1.5mm）         | color_banding:min_width=1.5mm     |

### 4. 致命错误预防（AI必须规避）
- 针迹碰撞：连续针迹角度变化≤120°（max_stitch_angle: 120_degree）
- 线张力过载：线长(m) = 1.3 × 位移距离(mm)（tension_comp: 1.3*√(dx²+dy²)）
- 累计误差溢出：每500针归零（auto_zeroing:every_500_stitches）

### 5. 增强AI工业兼容性的Prompt模板
```
生成符合Tajima DST 2.0规范的绣花文件：
1. 物理约束：
   - 最大单针位移：9.0mm（ΔX/ΔY=±127）
   - 最小针距：0.7mm（低于此值自动合并）
2. 生产安全：
   - 所有曲线应用Ramer-Douglas-Peucker简化（ε=0.3mm）
   - JUMP指令前强制插入TRIM（换色场景）
3. 效率优化：
   - 针迹路径应用贪心算法（总路径最短化）
   - 同色区块自动聚类刺绣（减少换色次数）
4. 输出校验：
   - 文件尾添加END(0x7)标志
   - 生成针迹统计表：总针数/各色耗线量(m)
```

### 6. 验证流程（伪代码/流程建议）
```python
# AI生成DST → 专业软件校验 → 上机试绣 → 错误分析与Prompt修正 → 断线率<0.1%量产批准

def ai_dst_generation_pipeline(image, prompt):
    dst_data = ai_model_generate_dst(image, prompt)
    if not verify_dst_constraints(dst_data):
        raise ValueError("DST文件不符合工业规范！")
    if not wilcom_es_software_check(dst_data):
        prompt = auto_fix_prompt(prompt, dst_data)
        return ai_dst_generation_pipeline(image, prompt)
    if not machine_test(dst_data):
        prompt = auto_fix_prompt(prompt, dst_data)
        return ai_dst_generation_pipeline(image, prompt)
    if get_break_rate(dst_data) >= 0.001:
        prompt = auto_fix_prompt(prompt, dst_data)
        return ai_dst_generation_pipeline(image, prompt)
    return dst_data
```

### 7. 校验与自动修正模块代码示例
```python
import math

def verify_stitch_constraints(stitches):
    total_x, total_y = 0, 0
    for i, s in enumerate(stitches):
        dx, dy = s['dx'], s['dy']
        # 单针最大位移
        if abs(dx) + abs(dy) > 256:
            return False, f"第{i}针位移超限"
        # 针迹碰撞
        if i > 1:
            prev = stitches[i-1]
            angle = math.degrees(math.atan2(dy, dx) - math.atan2(prev['dy'], prev['dx']))
            if abs(angle) > 120:
                return False, f"第{i}针角度变化过大"
        # 累计误差
        total_x += dx; total_y += dy
        if abs(total_x) > 4096 or abs(total_y) > 4096:
            return False, f"累计位移溢出"
    return True, "校验通过"

def auto_fix_prompt(prompt, dst_data):
    # 可根据校验结果自动调整Prompt参数
    # 例如：增加auto_zeroing、降低stitch_density等
    if "累计位移溢出" in dst_data['errors']:
        prompt += " auto_zeroing:every_500_stitches"
    if "位移超限" in dst_data['errors']:
        prompt += " max_stitch_move:256"
    if "角度变化过大" in dst_data['errors']:
        prompt += " max_stitch_angle:120_degree"
    return prompt
```

---

> **所有AI生成DST文件的训练、推理、验证、量产流程，必须100%遵循本章节工业强制标准。**

---

**文档版本：2.1**  
**更新日期：2023年10月15日**  
**编制单位：智能刺绣解决方案组**  
**许可协议：CC BY-SA 4.0** 