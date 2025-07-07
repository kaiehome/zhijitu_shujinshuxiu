import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Space,
  Typography,
  Slider,
  Switch,
  Row,
  Col,
  Alert,
  Divider,
  Tag,
  message,
  Radio,
  Checkbox,
} from 'antd';
// 使用emoji替代图标
import axios from 'axios';

const { Title, Text, Paragraph } = Typography;

// 数据格式转换函数：后端蛇形命名 → 前端驼峰命名
const transformBackendResponse = (backendData: any) => {
  if (!backendData) return null;
  
  return {
    jobId: backendData.job_id,
    status: backendData.status,
    message: backendData.message,
    originalFilename: backendData.original_filename,
    processedFiles: backendData.processed_files,
    processingTime: backendData.processing_time,
  };
};

interface ProcessSectionProps {
  uploadedFile: any;
  onProcessStart: (result: any) => void;
  onError?: (errorMessage: string) => void;
  disabled?: boolean;
  config?: any;
  onConfigChange?: (config: any) => void;
  processing?: boolean; // 添加外部处理状态
  processResult?: any; // 添加处理结果状态
}

const ProcessSection: React.FC<ProcessSectionProps> = ({ 
  uploadedFile, 
  onProcessStart, 
  onError,
  disabled = false,
  config,
  onConfigChange,
  processing: externalProcessing = false,
  processResult
}) => {
  const [localProcessing, setLocalProcessing] = useState(false);
  
  // 使用外部传入的processing状态，如果没有则使用本地状态
  const processing = externalProcessing || localProcessing;
  const [colorCount, setColorCount] = useState(config?.colorCount || 16);
  const [edgeEnhancement, setEdgeEnhancement] = useState(config?.edgeEnhancement ?? true);
  const [noiseReduction, setNoiseReduction] = useState(config?.noiseReduction ?? true);
  const [professionalMode, setProfessionalMode] = useState(config?.professionalMode ?? true);
  const [processingMode, setProcessingMode] = useState<'traditional' | 'structural'>('traditional');
  
  // 当外部processing状态变为false时，重置本地状态
  useEffect(() => {
    if (!externalProcessing) {
      setLocalProcessing(false);
    }
  }, [externalProcessing]);

  // 当处理成功后重置本地状态
  useEffect(() => {
    // 如果有处理结果且状态为完成或失败，重置本地处理状态
    if (processResult?.status === 'completed' || processResult?.status === 'failed') {
      setLocalProcessing(false);
    }
  }, [processResult]);

  const handleProcess = async () => {
    if (!uploadedFile) {
      message.error('请先上传图像');
      return;
    }

    setLocalProcessing(true);

    try {
      const formData = new FormData();
      // 使用uploadedFile.file (原始文件对象) 而不是整个uploadedFile对象
      formData.append('file', uploadedFile.file || uploadedFile);
      formData.append('color_count', colorCount.toString());
      formData.append('edge_enhancement', edgeEnhancement.toString());
      formData.append('noise_reduction', noiseReduction.toString());

      // 根据处理模式选择API端点
      const apiEndpoint = processingMode === 'structural' ? '/api/process-structural' : '/api/process';
      
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.text();
        let errorMessage = '处理失败';
        
        try {
          const errorJson = JSON.parse(errorData);
          errorMessage = errorJson.detail || errorMessage;
        } catch {
          errorMessage = errorData || errorMessage;
        }
        
        throw new Error(errorMessage);
      }

      const result = await response.json();
      
      // 转换数据格式
      const processedResult = {
        jobId: result.job_id,
        processingTime: result.processing_time,
        professionalImageUrl: result.professional_image_url,
        comparisonImageUrl: result.comparison_image_url,
        structureInfoUrl: result.structure_info_url, // 新增结构信息URL
        parameters: result.parameters,
        structureMetrics: result.structure_metrics, // 新增结构指标
        message: result.message,
        status: result.status
      };

      onProcessStart(processedResult);
      message.success(`${processingMode === 'structural' ? '结构化' : '传统'}处理完成！耗时 ${result.processing_time} 秒`);

    } catch (error: any) {
      console.error('处理失败:', error);
      message.error(error.message || '处理失败，请重试');
    } finally {
      setLocalProcessing(false);
    }
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }}>
      {/* 处理参数 */}
      <Card
        size="small"
        title={
          <Space>
            <span style={{ color: '#475569' }}>⚙️</span>
            <Text strong style={{ color: '#1e293b', fontWeight: 600, textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)' }}>处理参数</Text>
          </Space>
        }
        className="liquid-glass-card"
        style={{ 
          marginBottom: 16
        }}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          {/* 新增：处理模式选择 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              处理模式
            </label>
            <Radio.Group 
              value={processingMode} 
              onChange={(e) => setProcessingMode(e.target.value)}
              className="w-full"
            >
              <Space direction="vertical">
                <Radio value="traditional">
                  <div>
                    <div className="font-medium">传统模式</div>
                    <div className="text-sm text-gray-500">基于视觉效果优化，适合预览展示</div>
                  </div>
                </Radio>
                <Radio value="structural">
                  <div>
                    <div className="font-medium">结构化模式 🎯</div>
                    <div className="text-sm text-gray-500">
                      基于机器识别优化，生成具备结构特征的专业识别图
                    </div>
                  </div>
                </Radio>
              </Space>
            </Radio.Group>
          </div>
          
          {/* 颜色数量 */}
          <Row align="middle" gutter={[16, 0]}>
            <Col span={6}>
              <Space>
                <span style={{ color: '#475569' }}>🎨</span>
                <Text strong style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>颜色数量</Text>
              </Space>
            </Col>
            <Col span={15}>
              <div className="liquid-progress" style={{ padding: '0 20px' }}>
                <Slider
                  min={10}
                  max={20}
                  step={null}
                  value={colorCount}
                  onChange={(value) => {
                    setColorCount(value);
                    onConfigChange?.({ colorCount: value });
                  }}
                  marks={{
                    10: { style: { color: '#1e293b', fontWeight: 500, textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }, label: '10色' },
                    12: { style: { color: '#1e293b', fontWeight: 500, textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }, label: '12色' },
                    14: { style: { color: '#1e293b', fontWeight: 500, textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }, label: '14色' },
                    16: { style: { color: '#1e293b', fontWeight: 500, textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }, label: '16色' },
                                          18: { style: { color: '#1e293b', fontWeight: 500, textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }, label: '18色' },
                    20: { style: { color: '#1e293b', fontWeight: 500, textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }, label: '20色' },
                  }}
                  included={false}
                  disabled={disabled}
                />
              </div>
            </Col>
            <Col span={3} style={{ textAlign: 'right' }}>
              <Tag className="liquid-tag" style={{ 
                color: '#1e293b',
                fontWeight: 600,
                textShadow: '0 1px 1px rgba(255, 255, 255, 0.5)'
              }}>
                {colorCount}色
              </Tag>
            </Col>
          </Row>
          
          <Divider style={{ margin: '12px 0', borderColor: 'rgba(148, 163, 184, 0.3)' }} />
          
          {/* 边缘增强 */}
          <Row align="middle">
            <Col span={8}>
              <Space>
                <span style={{ color: '#475569' }}>✨</span>
                <Text strong style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>边缘增强</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Text style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>增强图案轮廓，提升识别度</Text>
            </Col>
            <Col span={4} style={{ textAlign: 'right' }}>
              <Switch
                checked={edgeEnhancement}
                onChange={(checked) => {
                  setEdgeEnhancement(checked);
                  onConfigChange?.({ edgeEnhancement: checked });
                }}
                disabled={disabled}
                className="liquid-switch"
              />
            </Col>
          </Row>
          
          <Divider style={{ margin: '12px 0', borderColor: 'rgba(148, 163, 184, 0.3)' }} />
          
          {/* 噪声清理 */}
          <Row align="middle">
            <Col span={8}>
              <Space>
                <span style={{ color: '#475569' }}>🧹</span>
                <Text strong style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>噪声清理</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Text style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>清理图像噪点，平滑处理</Text>
            </Col>
            <Col span={4} style={{ textAlign: 'right' }}>
              <Switch
                checked={noiseReduction}
                onChange={(checked) => {
                  setNoiseReduction(checked);
                  onConfigChange?.({ noiseReduction: checked });
                }}
                disabled={disabled}
                className="liquid-switch"
              />
            </Col>
          </Row>
          
          <Divider style={{ margin: '12px 0', borderColor: 'rgba(148, 163, 184, 0.3)' }} />
          
          {/* 专业织机模式 */}
          <Row align="middle">
            <Col span={8}>
              <Space>
                <span style={{ color: '#475569' }}>🏭</span>
                <Text strong style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>专业织机模式</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Text style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>专业级处理，色块连贯，边框装饰</Text>
            </Col>
            <Col span={4} style={{ textAlign: 'right' }}>
              <Switch
                checked={professionalMode}
                onChange={(checked) => {
                  setProfessionalMode(checked);
                  onConfigChange?.({ professionalMode: checked });
                }}
                disabled={disabled}
                className="liquid-switch"
              />
            </Col>
          </Row>
        </Space>
      </Card>

      {/* 处理说明 */}
      <Alert
        message={
          <span style={{ 
            color: '#1e293b', 
            fontWeight: 600,
            textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)'
          }}>
            蜀锦蜀绣风格处理
          </span>
        }
        description={
          <Space direction="vertical">
            <Text style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>
              • <strong style={{ color: '#0f172a', textShadow: '0 1px 1px rgba(255, 255, 255, 0.8)' }}>颜色简化：</strong>使用K-means聚类算法，将图像颜色降至{colorCount}种主要色彩
            </Text>
            <Text style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>
              • <strong style={{ color: '#0f172a', textShadow: '0 1px 1px rgba(255, 255, 255, 0.8)' }}>风格优化：</strong>自动调整饱和度、对比度，符合传统蜀锦审美
            </Text>
            <Text style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>
              • <strong style={{ color: '#0f172a', textShadow: '0 1px 1px rgba(255, 255, 255, 0.8)' }}>织机适配：</strong>优化边缘清晰度，确保织机准确识别
            </Text>
            {professionalMode && (
              <Text style={{ color: '#1e293b', textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' }}>
                • <strong style={{ color: '#d97706', textShadow: '0 1px 1px rgba(255, 255, 255, 0.8)' }}>专业织机模式：</strong>色块连贯性增强，专业装饰边框，对比图生成
              </Text>
            )}
          </Space>
        }
        type="info"
        showIcon
        style={{ 
          background: 'rgba(255, 255, 255, 0.2)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(148, 163, 184, 0.3)',
          borderRadius: '12px',
          marginBottom: 16
        }}
      />

      {/* 开始处理按钮 */}
      <Button
        type="primary"
        size="large"
        block
        icon={<span style={{ marginRight: '8px' }}>▶️</span>}
        onClick={handleProcess}
        loading={processing}
        disabled={disabled || processing}
        className={`liquid-button liquid-ripple ${processing ? 'liquid-pulse' : ''}`}
        style={{
          height: 50,
          fontSize: 16,
          fontWeight: 600,
        }}
      >
        {processing ? '正在处理中...' : (professionalMode ? '🏭 生成专业织机图' : '开始生成打样图')}
      </Button>

      {/* 处理模式说明 */}
      {processingMode === 'structural' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start space-x-2">
            <span className="text-blue-600 text-lg">🎯</span>
            <div>
              <div className="font-medium text-blue-800 mb-1">结构化专业模式特点：</div>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• 区域连续性：每个色块为闭合区域，可用于填充</li>
                <li>• 边界清晰性：路径可提取、点数可控</li>
                <li>• 颜色可控性：避免相邻干扰、符合绣线规范</li>
                <li>• 机器可读：生成矢量路径和结构信息</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </Space>
  );
};

export default ProcessSection; 