import React, { useEffect, useState } from 'react';
import {
  Space,
  Button,
  Typography,
  Card,
  Row,
  Col,
  Alert,
  Progress,
  Image,
  Tag,
  Spin,
  message,
  Divider,
  Slider,
} from 'antd';
import DownloadOutlined from '@ant-design/icons/DownloadOutlined';
import CheckCircleOutlined from '@ant-design/icons/CheckCircleOutlined';
import CloseCircleOutlined from '@ant-design/icons/CloseCircleOutlined';
import SyncOutlined from '@ant-design/icons/SyncOutlined';
import ReloadOutlined from '@ant-design/icons/ReloadOutlined';
import FileImageOutlined from '@ant-design/icons/FileImageOutlined';
import FileTextOutlined from '@ant-design/icons/FileTextOutlined';
import axios from 'axios';

const { Title, Text, Paragraph } = Typography;

interface ResultSectionProps {
  processResult: any;
  onProcessUpdate: (result: any) => void;
  onError?: (errorMessage: string) => void;
  onReset: () => void;
  originalImageUrl: string;
  professionalImageUrl: string;
  imageWidth: number;
  imageHeight: number;
  keypoints?: any[];
  regions?: any[];
  structureInfoUrl?: string;
  loading?: boolean;
  colorCount: number;
  onColorCountChange: (val: number) => void;
  onReprocess: () => void;
}

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

const colorMarks = {10:'10色',12:'12色',14:'14色',16:'16色',18:'18色',20:'20色'};

const ResultSection: React.FC<ResultSectionProps> = ({ 
  processResult, 
  onProcessUpdate, 
  onError,
  onReset,
  originalImageUrl,
  professionalImageUrl,
  imageWidth,
  imageHeight,
  keypoints,
  regions,
  structureInfoUrl,
  loading,
  colorCount,
  onColorCountChange,
  onReprocess,
}) => {
  const [polling, setPolling] = useState(false);
  const [progressValue, setProgressValue] = useState(0);
  const [pollingDuration, setPollingDuration] = useState(0);
  const [notified, setNotified] = useState(false); // 防止重复通知
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (processResult?.status === 'processing') {
      pollStatus();
    }
  }, [processResult]);

  // 计算动态进度
  const calculateProgress = (duration: number) => {
    // 前10秒快速增长到30%
    if (duration <= 10) {
      return Math.min(30, duration * 3);
    }
    // 10-30秒缓慢增长到70%
    else if (duration <= 30) {
      return 30 + (duration - 10) * 2;
    }
    // 30-60秒缓慢增长到90%
    else if (duration <= 60) {
      return 70 + (duration - 30) * 0.67;
    }
    // 60秒后保持90%
    else {
      return 90;
    }
  };

  const pollStatus = async () => {
    const jobId = processResult?.jobId || processResult?.job_id;
    if (!jobId) {
      console.error('轮询失败：缺少任务ID');
      onError?.('任务ID丢失，请重新开始');
      return;
    }
    
    setPolling(true);
    setPollingDuration(0);
    setProgressValue(0);
    setNotified(false); // 重置通知状态
    
    const startTime = Date.now();
    
    const pollInterval = setInterval(async () => {
      try {
        const currentDuration = Math.floor((Date.now() - startTime) / 1000);
        setPollingDuration(currentDuration);
        setProgressValue(calculateProgress(currentDuration));
        
        const response = await axios.get(`/api/status/${jobId}`);
        const backendData = response.data;
        
        // 转换数据格式
        const transformedResult = transformBackendResponse(backendData);
        
        if (!transformedResult) {
          console.error('数据转换失败:', backendData);
          onError?.('数据格式错误，请重试');
          clearInterval(pollInterval);
          setPolling(false);
          return;
        }
        
        console.log('轮询状态:', transformedResult);
        onProcessUpdate(transformedResult);
        
        if (transformedResult.status === 'completed' || transformedResult.status === 'failed') {
          clearInterval(pollInterval);
          setPolling(false);
          setProgressValue(100);
          
          // 只在第一次检测到完成/失败状态时才发送通知
          if (!notified) {
            setNotified(true);
            // 移除这里的通知，统一由主页面处理，避免重复
            // if (transformedResult.status === 'completed') {
            //   message.success('图像处理完成！');
            // } else {
            //   message.error('图像处理失败');
            // }
          }
        }
      } catch (error) {
        console.error('轮询状态失败:', error);
        const errorMessage = '轮询状态失败，请重试';
        onError?.(errorMessage);
        clearInterval(pollInterval);
        setPolling(false);
      }
    }, 2000);

    // 60秒后停止轮询
    setTimeout(() => {
      clearInterval(pollInterval);
      setPolling(false);
    }, 60000);
  };

  const downloadFile = async (jobId: string, filename: string, displayName: string) => {
    try {
      console.log('开始下载文件:', { jobId, filename, displayName });
      
      // 显示下载提示
      message.loading(`正在下载 ${displayName}...`, 0);
      
      const response = await axios.get(`/api/download/${jobId}/${filename}`, {
        responseType: 'blob',
        timeout: 30000, // 30秒超时
      });
      
      console.log('下载响应:', response.status, response.headers);
      
      // 检查响应状态
      if (response.status !== 200) {
        throw new Error(`下载失败，状态码: ${response.status}`);
      }
      
      // 检查响应数据
      if (!response.data || response.data.size === 0) {
        throw new Error('下载的文件为空');
      }
      
      console.log('文件大小:', response.data.size, 'bytes');
      
      // 创建blob和下载链接
      const blob = new Blob([response.data], { 
        type: response.headers['content-type'] || 'application/octet-stream' 
      });
      
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = displayName; // 使用download属性而不是setAttribute
      link.style.display = 'none';
      
      // 添加到DOM，点击，然后移除
      document.body.appendChild(link);
      link.click();
      
      // 延迟清理，确保下载开始
      setTimeout(() => {
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      }, 100);
      
      // 清除loading提示
      message.destroy();
      message.success(`${displayName} 下载成功！`);
      
    } catch (error: any) {
      console.error('下载失败:', error);
      
      // 清除loading提示
      message.destroy();
      
      let errorMessage = '下载失败，请重试';
      if (error.code === 'ECONNABORTED') {
        errorMessage = '下载超时，请检查网络连接';
      } else if (error.response) {
        errorMessage = `下载失败：${error.response.status} ${error.response.statusText}`;
      } else if (error.message) {
        errorMessage = `下载失败：${error.message}`;
      }
      
      message.error(errorMessage);
      onError?.(errorMessage);
    }
  };

  const getProgressStatus = () => {
    if (progressValue >= 90) return 'normal';
    if (progressValue >= 70) return 'active';
    return 'active';
  };

  const getProgressMessage = () => {
    if (pollingDuration <= 10) return '正在分析图像结构...';
    if (pollingDuration <= 30) return '正在进行颜色聚类分析...';
    if (pollingDuration <= 45) return '正在生成高清打样图...';
    if (pollingDuration <= 60) return '正在生成矢量辅助图...';
    return '即将完成处理...';
  };

  const renderProcessingStatus = () => (
    <Card
      style={{
        background: 'rgba(250, 173, 20, 0.1)',
        border: '2px solid #faad14',
        borderRadius: 8,
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }} align="center">
        <Spin size="large" />
        <Title level={4}>AI正在处理中...</Title>
        <Progress 
          percent={Math.round(progressValue)} 
          status={getProgressStatus()}
          strokeColor="#faad14" 
        />
        <Text type="secondary">
          任务ID: {processResult?.jobId || processResult?.job_id || '获取中...'}
        </Text>
        <Divider style={{ margin: '16px 0' }} />
        <Text type="secondary" style={{ textAlign: 'center', fontSize: '14px' }}>
          {getProgressMessage()}
        </Text>
        <Text type="secondary" style={{ textAlign: 'center', fontSize: '12px' }}>
          已处理 {pollingDuration} 秒 • 预计还需 {Math.max(0, 45 - pollingDuration)} 秒
        </Text>
      </Space>
    </Card>
  );

  const renderCompletedStatus = () => (
    <Space direction="vertical" style={{ width: '100%' }}>
      <Card
        style={{
          background: 'rgba(82, 196, 26, 0.1)',
          border: '2px solid #52c41a',
          borderRadius: 8,
        }}
      >
        <Space direction="vertical" style={{ width: '100%' }} align="center">
          <CheckCircleOutlined style={{ fontSize: 48, color: '#52c41a' }} />
          <Title level={4}>处理完成！</Title>
          <Space wrap>
            <Tag color="success">
              处理时间: {processResult.processingTime?.toFixed(2)}秒
            </Tag>
            <Tag color="blue">
              任务ID: {processResult.jobId}
            </Tag>
            {processResult.parameters?.generator_type && (
              <Tag color={processResult.parameters.generator_type === 'structural_professional' ? 'purple' : 'green'}>
                {processResult.parameters.generator_type === 'structural_professional' ? '🎯 结构化模式' : '传统模式'}
              </Tag>
            )}
          </Space>
        </Space>
      </Card>

      {/* 结构化模式特有的结构指标 */}
      {processResult.structureMetrics && (
        <Card title="🔧 结构化分析结果" size="small">
          <div style={{ background: 'rgba(24, 144, 255, 0.1)', padding: 16, borderRadius: 8 }}>
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={6}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
                    {processResult.structureMetrics.total_regions}
                  </div>
                  <div style={{ fontSize: 12, color: '#666' }}>闭合区域</div>
                </div>
              </Col>
              <Col xs={12} sm={6}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
                    {processResult.structureMetrics.total_boundaries}
                  </div>
                  <div style={{ fontSize: 12, color: '#666' }}>矢量边界</div>
                </div>
              </Col>
              <Col xs={12} sm={6}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
                    {processResult.structureMetrics.color_palette_size}
                  </div>
                  <div style={{ fontSize: 12, color: '#666' }}>颜色数量</div>
                </div>
              </Col>
              <Col xs={12} sm={6}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>✓</div>
                  <div style={{ fontSize: 12, color: '#666' }}>机器可读</div>
                </div>
              </Col>
            </Row>
            
            <div style={{ marginTop: 12 }}>
              <Space wrap>
                {processResult.structureMetrics.is_machine_readable && (
                  <Tag color="green">机器可读</Tag>
                )}
                {processResult.structureMetrics.has_vector_paths && (
                  <Tag color="blue">矢量路径</Tag>
                )}
                {processResult.structureMetrics.region_closure_validated && (
                  <Tag color="purple">区域闭合验证</Tag>
                )}
              </Space>
            </div>
          </div>
        </Card>
      )}

      {/* 图像预览区域 */}
      {(processResult.professionalImage || processResult.professionalImageUrl || processResult.comparisonImage || processResult.comparisonImageUrl) && (
        <Card title="图像预览" size="small">
          <Row gutter={[16, 16]}>
            {(processResult.professionalImage || processResult.professionalImageUrl) && (
              <Col xs={24} md={12}>
                <Card title="织机识别图" bordered={false}>
                  <div style={{ position: 'relative', width: '100%' }}>
                    <Image
                      src={processResult.professionalImageUrl || `/${processResult.professionalImage}`}
                      alt="专业识别图"
                      style={{ width: '100%', display: 'block' }}
                    />
                  </div>
                  <Space style={{ marginTop: 16 }}>
                    <Button
                      type="primary"
                      icon={<DownloadOutlined />}
                      loading={downloading}
                      onClick={() => downloadFile(processResult.jobId, processResult.professionalImageUrl?.split('/').pop() || '', '织机识别图.png')}
                    >
                      下载识别图
                    </Button>
                    {structureInfoUrl && (
                      <Button
                        icon={<DownloadOutlined />}
                        onClick={() => downloadFile(processResult.jobId, structureInfoUrl?.split('/').pop() || '', '结构信息.json')}
                      >
                        下载结构信息
                      </Button>
                    )}
                  </Space>
                </Card>
              </Col>
            )}
            {(processResult.comparisonImage || processResult.comparisonImageUrl) && (
              <Col xs={24} md={12}>
                <Card size="small" title="处理前后对比图">
                  <Image
                    src={processResult.comparisonImageUrl || `/${processResult.comparisonImage}`}
                    alt="处理前后对比图"
                    style={{ width: '100%' }}
                  />
                  <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>原图与处理结果对比</Text>
                    <Button 
                      size="small" 
                      onClick={() => window.open(processResult.comparisonImageUrl || `/${processResult.comparisonImage}`, '_blank')}
                    >
                      下载
                    </Button>
                  </div>
                </Card>
              </Col>
            )}
          </Row>
        </Card>
      )}

      {/* 文件下载区域 */}
      <Card title="下载文件" size="small">
        <Row gutter={[16, 16]}>
          {processResult.processedFiles?.map((filePath: string, index: number) => {
            const filename = filePath.split('/').pop() || '';
            const isPNG = filename.endsWith('.png');
            const isComparison = filename.includes('comparison');
            let displayName = '';
            let description = '';
            
            if (isPNG && isComparison) {
              displayName = '处理前后对比图.png';
              description = '原图与处理结果对比';
            } else if (isPNG) {
              displayName = '专业织机识别图.png';
              description = '织机识别主图';
            } else {
              displayName = '辅助矢量图.svg';
              description = '矢量辅助文件';
            }
            
            return (
              <Col xs={24} sm={12} key={index}>
                <Card
                  size="small"
                  style={{
                    background: isPNG ? 'rgba(24, 144, 255, 0.1)' : 'rgba(135, 208, 104, 0.1)',
                    border: `1px solid ${isPNG ? '#1890ff' : '#87d068'}`,
                  }}
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Space>
                      {isPNG ? (
                        <FileImageOutlined style={{ color: '#1890ff', fontSize: 20 }} />
                      ) : (
                        <FileTextOutlined style={{ color: '#87d068', fontSize: 20 }} />
                      )}
                      <div>
                        <Text strong>{displayName}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {description}
                        </Text>
                      </div>
                    </Space>
                    <Button
                      type="primary"
                      block
                      icon={<DownloadOutlined />}
                      onClick={() => downloadFile(processResult.jobId, filename, displayName)}
                      style={{
                        background: isPNG ? '#1890ff' : '#87d068',
                        border: 'none',
                      }}
                    >
                      下载
                    </Button>
                  </Space>
                </Card>
              </Col>
            );
          })}
        </Row>
      </Card>

      {/* 使用说明 */}
      <Alert
        message="文件说明"
        description={
          <div>
            <Paragraph>
              • <strong>高清打样图(PNG)：</strong>已优化的高清图像，可直接用于织机识别和刺绣制作
            </Paragraph>
            <Paragraph style={{ marginBottom: 0 }}>
              • <strong>辅助矢量图(SVG)：</strong>矢量格式文件，便于后期编辑和路径调整
            </Paragraph>
          </div>
        }
        type="success"
        showIcon
        style={{ background: 'rgba(82, 196, 26, 0.1)' }}
      />
    </Space>
  );

  const renderFailedStatus = () => (
    <Card
      style={{
        background: 'rgba(255, 77, 79, 0.1)',
        border: '2px solid #ff4d4f',
        borderRadius: 8,
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }} align="center">
        <CloseCircleOutlined style={{ fontSize: 48, color: '#ff4d4f' }} />
        <Title level={4}>处理失败</Title>
        <Text type="danger">{processResult.message}</Text>
        <Button
          type="primary"
          danger
          icon={<ReloadOutlined />}
          onClick={onReset}
        >
          重新开始
        </Button>
      </Space>
    </Card>
  );

  const renderContent = () => {
    switch (processResult.status) {
      case 'processing':
        return renderProcessingStatus();
      case 'completed':
        return renderCompletedStatus();
      case 'failed':
        return renderFailedStatus();
      default:
        return null;
    }
  };

  const handleDownload = async (url: string, filename: string) => {
    try {
      setDownloading(true);
      const res = await fetch(url);
      const blob = await res.blob();
      const link = document.createElement('a');
      link.href = window.URL.createObjectURL(blob);
      link.download = filename;
      link.click();
      setTimeout(() => window.URL.revokeObjectURL(link.href), 2000);
      setDownloading(false);
    } catch (e) {
      setDownloading(false);
      message.error('下载失败');
    }
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }}>
      {renderContent()}
      
      {processResult.status === 'completed' && (
        <>
          <Divider />
          <Button
            block
            icon={<ReloadOutlined />}
            onClick={onReset}
            style={{ marginTop: 16 }}
          >
            处理新图像
          </Button>
        </>
      )}

      {/* 参数调整区 */}
      <Card
        title={<Space><ReloadOutlined />参数调整</Space>}
        style={{ marginTop: 24 }}
        bordered={false}
      >
        <Row align="middle" gutter={24}>
          <Col xs={24} md={12}>
            <Text>颜色数量：</Text>
            <Slider
              min={10}
              max={20}
              step={2}
              marks={colorMarks}
              value={colorCount}
              onChange={onColorCountChange}
              style={{ width: 200, display: 'inline-block', marginLeft: 16 }}
              disabled={loading}
            />
          </Col>
          <Col xs={24} md={12}>
            <Button
              type="primary"
              icon={<ReloadOutlined />}
              onClick={onReprocess}
              disabled={loading}
            >
              重新生成
            </Button>
          </Col>
        </Row>
      </Card>
    </Space>
  );
};

export default ResultSection; 