/**
 * 蜀锦蜀绣AI打样图生成工具 - 主页面
 * 提供完整的图像处理工作流程：上传 -> 处理 -> 结果展示
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  Layout,
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Alert,
  Divider,
  message,
  Steps,
  Spin,
  Result,
  Tooltip,
  Progress,
} from 'antd';
import CloudUploadOutlined from '@ant-design/icons/CloudUploadOutlined';
import SettingOutlined from '@ant-design/icons/SettingOutlined';
import DownloadOutlined from '@ant-design/icons/DownloadOutlined';
import ReloadOutlined from '@ant-design/icons/ReloadOutlined';
import InfoCircleOutlined from '@ant-design/icons/InfoCircleOutlined';

import Header from '../components/Header';
import UploadSection from '../components/UploadSection';
import ProcessSection from '../components/ProcessSection';
import ResultSection from '../components/ResultSection';

const { Content } = Layout;
const { Title, Paragraph, Text } = Typography;
const { Step } = Steps;

// 类型定义
export interface ProcessResult {
  success?: boolean;
  jobId: string;
  status: 'processing' | 'completed' | 'failed' | 'pending';
  message: string;
  originalFilename?: string;
  processedFiles?: string[];
  processingTime?: number;
  professionalImage?: string;
  comparisonImage?: string;
}

export interface UploadedFile {
  filename: string;
  size: number;
  contentType: string;
  uploadTime: string;
}

export interface ProcessConfig {
  colorCount: number;
  edgeEnhancement: boolean;
  noiseReduction: boolean;
  style: string;
}

// 常量定义
const PROCESSING_STEPS = [
  { title: '上传图片', icon: <CloudUploadOutlined /> },
  { title: '配置处理', icon: <SettingOutlined /> },
  { title: '查看结果', icon: <DownloadOutlined /> }
];

const FEATURE_CARDS = [
  {
    id: 'upload',
    icon: '☁️',
    title: '智能上传',
    description: '支持JPG/PNG格式，最大10MB',
    color: 'rgba(248, 113, 113, 0.2)',
    borderColor: 'rgba(248, 113, 113, 0.3)',
    titleColor: '#dc2626',
    textColor: '#7f1d1d'
  },
  {
    id: 'process',
    icon: '🎨',
    title: 'AI处理',
    description: '颜色降色、边缘增强、风格化',
    color: 'rgba(96, 165, 250, 0.2)',
    borderColor: 'rgba(96, 165, 250, 0.3)',
    titleColor: '#2563eb',
    textColor: '#1e3a8a'
  },
  {
    id: 'output',
    icon: '📥',
    title: '高清输出',
    description: 'PNG主图+SVG辅助文件',
    color: 'rgba(139, 92, 246, 0.2)',
    borderColor: 'rgba(139, 92, 246, 0.3)',
    titleColor: '#7c3aed',
    textColor: '#581c87'
  },
  {
    id: 'tradition',
    icon: '🏮',
    title: '传统工艺',
    description: '专注蜀锦蜀绣传统风格',
    color: 'rgba(74, 222, 128, 0.2)',
    borderColor: 'rgba(74, 222, 128, 0.3)',
    titleColor: '#16a34a',
    textColor: '#14532d'
  }
];

export default function Home() {
  // 状态管理
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [processResult, setProcessResult] = useState<ProcessResult | null>(null);
  const [currentStep, setCurrentStep] = useState<'upload' | 'process' | 'result'>('upload');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processConfig, setProcessConfig] = useState<ProcessConfig>({
    colorCount: 16,
    edgeEnhancement: true,
    noiseReduction: true,
    style: 'sichuan_brocade'
  });

  // 计算当前步骤索引
  const currentStepIndex = useMemo(() => {
    switch (currentStep) {
      case 'upload': return 0;
      case 'process': return 1;
      case 'result': return 2;
      default: return 0;
    }
  }, [currentStep]);

  // 错误处理
  const handleError = useCallback((errorMessage: string) => {
    setError(errorMessage);
    setLoading(false);
    message.error(errorMessage);
  }, []);

  // 清除错误
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // 上传成功处理
  const handleUploadSuccess = useCallback((file: UploadedFile) => {
    try {
      setUploadedFile(file);
      setCurrentStep('process');
      setProcessResult(null);
      clearError();
      
      message.success(`文件上传成功: ${file.filename}`);
    } catch (err) {
      handleError('处理上传结果时发生错误');
    }
  }, [clearError, handleError]);

  // 处理开始
  const handleProcessStart = useCallback((result: ProcessResult) => {
    try {
      setProcessResult(result);
      setCurrentStep('result');
      
      // 如果后端直接返回完成状态，不需要loading
      if (result.status === 'completed') {
        setLoading(false);
        message.success(`处理完成！耗时: ${result.processingTime?.toFixed(2)}秒`);
      } else {
        setLoading(true);
        message.info('图像处理已开始，请稍候...');
      }
      
      clearError();
    } catch (err) {
      handleError('启动处理任务时发生错误');
    }
  }, [clearError, handleError]);

  // 处理状态更新 - 增加防重复通知
  const [lastNotifiedStatus, setLastNotifiedStatus] = useState<string | null>(null);
  
  const handleProcessUpdate = useCallback((result: ProcessResult) => {
    try {
      setProcessResult(result);
      
      // 防止重复通知：只在状态首次变为完成/失败时通知
      const statusKey = `${result.jobId}-${result.status}`;
      if (result.status === 'completed' && lastNotifiedStatus !== statusKey) {
        setLoading(false);
        setLastNotifiedStatus(statusKey);
        message.success(`处理完成！耗时: ${result.processingTime?.toFixed(2)}秒`);
      } else if (result.status === 'failed' && lastNotifiedStatus !== statusKey) {
        setLoading(false);
        setLastNotifiedStatus(statusKey);
        handleError(result.message || '图像处理失败');
      } else if (result.status === 'completed' || result.status === 'failed') {
        // 状态已变为完成/失败，但不重复通知
        setLoading(false);
      }
    } catch (err) {
      handleError('更新处理状态时发生错误');
    }
  }, [handleError, lastNotifiedStatus]);

  // 重置所有状态
  const handleReset = useCallback(() => {
    setUploadedFile(null);
    setProcessResult(null);
    setCurrentStep('upload');
    setLoading(false);
    setLastNotifiedStatus(null); // 重置通知状态
    clearError();
    
    message.info('已重置，可以开始新的处理任务');
  }, [clearError]);

  // 配置更新
  const handleConfigChange = useCallback((config: Partial<ProcessConfig>) => {
    setProcessConfig(prev => ({ ...prev, ...config }));
  }, []);

  // 页面标题
  const pageTitle = useMemo(() => {
    return (
      <Title 
        level={1} 
        style={{ 
          marginBottom: 16, 
          fontWeight: 600,
          fontSize: '3rem',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, "Noto Sans", sans-serif',
          letterSpacing: '1px',
          color: '#1e293b'
        }}
      >
        🧵✨ 蜀锦蜀绣 AI 打样图生成工具
      </Title>
    );
  }, []);

  // 功能卡片渲染 - 优化性能，减少inline样式和事件处理
  const renderFeatureCard = useCallback((feature: typeof FEATURE_CARDS[0], index: number) => (
    <Col xs={24} sm={12} md={6} key={feature.id}>
      <div 
        className={`feature-card feature-card-${index + 1}`} 
        style={{ 
          textAlign: 'center', 
          padding: '20px', 
          borderRadius: '16px',
          cursor: 'pointer'
        }}
      >
        <div 
          className="liquid-glow" 
          style={{ 
            background: `linear-gradient(135deg, ${feature.color}, ${feature.color.replace('0.2', '0.15')})`, 
            borderRadius: '50%', 
            width: '80px', 
            height: '80px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            margin: '0 auto 16px',
            backdropFilter: 'blur(15px)',
            border: `1px solid ${feature.borderColor}`,
            transition: 'all 0.3s ease'
          }}
        >
          <span style={{ fontSize: 32 }}>{feature.icon}</span>
        </div>
        <Title 
          level={4} 
          style={{ 
            color: feature.titleColor, 
            marginBottom: 8, 
            fontWeight: 600, 
            textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)' 
          }}
        >
          {feature.title}
        </Title>
        <Paragraph 
          style={{ 
            color: feature.textColor, 
            marginBottom: 0, 
            textShadow: '0 1px 1px rgba(255, 255, 255, 0.6)' 
          }}
        >
          {feature.description}
        </Paragraph>
      </div>
    </Col>
  ), []);

  // 步骤指示器
  const stepIndicator = useMemo(() => (
    <Card
      className="liquid-glass-card liquid-fade-in"
      style={{ marginBottom: 24 }}
    >
      <Steps 
        current={currentStepIndex} 
        size="small"
        items={PROCESSING_STEPS.map((step, index) => ({
          title: step.title,
          icon: step.icon,
          status: index < currentStepIndex ? 'finish' : 
                 index === currentStepIndex ? (loading ? 'process' : 'wait') : 'wait'
        }))}
      />
    </Card>
  ), [currentStepIndex, loading]);

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header />
      
      <Content style={{ padding: '24px 50px' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          
          {/* 错误提示 */}
          {error && (
            <Alert
              message="操作失败"
              description={error}
              type="error"
              showIcon
              closable
              onClose={clearError}
              style={{ marginBottom: 24 }}
              action={
                <Button size="small" danger onClick={handleReset}>
                  重新开始
                </Button>
              }
            />
          )}

          {/* 标题区域 */}
          <Card
            className="liquid-glass-card liquid-fade-in liquid-float"
            style={{
              marginBottom: 24,
              textAlign: 'center',
            }}
          >
            <Row justify="center" align="middle">
              <Col span={24}>
                {pageTitle}
                <Paragraph style={{ 
                  fontSize: 18, 
                  marginBottom: 0, 
                  lineHeight: '1.8',
                  color: '#475569',
                  textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)'
                }}>
                  专业的织机识别图像处理工具，传承千年蜀锦工艺，融合现代AI技术
                </Paragraph>
              </Col>
            </Row>
          </Card>

          {/* 功能介绍卡片 */}
          <Card
            className="liquid-glass-card liquid-fade-in liquid-float-delayed"
            style={{ marginBottom: 24 }}
          >
            <Row gutter={[24, 24]} justify="center">
              {FEATURE_CARDS.map(renderFeatureCard)}
            </Row>
          </Card>

          {/* 上传区域 */}
          <Card
            title={
              <Space>
                <CloudUploadOutlined style={{ color: '#1f2937' }} />
                <span style={{ 
                  color: '#1f2937', 
                  fontWeight: 600,
                  fontSize: '16px',
                  textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)'
                }}>
                  上传图片
                </span>
                {uploadedFile && (
                  <Tooltip title="已上传文件">
                    <Text type="success">✓</Text>
                  </Tooltip>
                )}
              </Space>
            }
            className="liquid-glass-card liquid-fade-in"
            style={{ 
              marginBottom: 24,
              minHeight: '256px'
            }}
            extra={
              uploadedFile && currentStep !== 'upload' && (
                <Button
                  type="link"
                  size="small"
                  onClick={() => {
                    setUploadedFile(null);
                    setProcessResult(null);
                    setCurrentStep('upload');
                    setLoading(false);
                    clearError();
                    message.info('已清除上传文件，可以重新上传');
                  }}
                >
                  重新上传
                </Button>
              )
            }
          >
            <UploadSection
              onUploadSuccess={handleUploadSuccess}
              onError={handleError}
              disabled={loading}
              uploadedFile={uploadedFile}
            />
          </Card>

          {/* 步骤指示器 */}
          {stepIndicator}

          {/* 处理配置区域 */}
          {uploadedFile && (
            <Card
              title={
                <Space>
                  <SettingOutlined style={{ color: '#1f2937' }} />
                  <span style={{ 
                    color: '#1f2937', 
                    fontWeight: 600,
                    fontSize: '16px',
                    textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)'
                  }}>
                    处理配置
                  </span>
                  {processResult && processResult.status === 'processing' && (
                    <Tooltip title="处理中">
                      <Spin size="small" />
                    </Tooltip>
                  )}
                </Space>
              }
              className="liquid-glass-card liquid-fade-in"
              style={{
                marginBottom: 24
              }}
              extra={
                processResult && currentStep !== 'process' && (
                  <Button
                    type="link"
                    size="small"
                    onClick={() => setCurrentStep('process')}
                  >
                    重新配置
                  </Button>
                )
              }
            >
              <ProcessSection
                uploadedFile={uploadedFile}
                onProcessStart={handleProcessStart}
                onError={handleError}
                disabled={loading}
                config={processConfig}
                onConfigChange={handleConfigChange}
                processing={loading}
                processResult={processResult}
              />
            </Card>
          )}

          {/* 处理结果区域 */}
          {processResult && (
            <Card
              title={
                <Space>
                  <DownloadOutlined style={{ color: '#1f2937' }} />
                  <span style={{ 
                    color: '#1f2937', 
                    fontWeight: 600,
                    fontSize: '16px',
                    textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)'
                  }}>
                    处理结果
                  </span>
                  {loading && (
                    <Tooltip title="处理中">
                      <Spin size="small" />
                    </Tooltip>
                  )}
                </Space>
              }
              className="liquid-glass-card liquid-fade-in"
              style={{
                marginBottom: 24,
                minHeight: '400px'
              }}
              extra={
                <Space>
                  {processResult.status === 'completed' && (
                    <Button
                      type="primary"
                      icon={<ReloadOutlined />}
                      onClick={handleReset}
                      size="small"
                    >
                      处理新图片
                    </Button>
                  )}
                </Space>
              }
            >
              <ResultSection
                processResult={processResult}
                onProcessUpdate={handleProcessUpdate}
                onError={handleError}
                onReset={handleReset}
              />
            </Card>
          )}

          {/* 主要内容区域 */}
          <Row gutter={[24, 24]}>
            {/* 使用提示 */}
            <Col xs={24}>
              <Card
                className="liquid-glass-card liquid-fade-in liquid-float-delayed"
                style={{ marginBottom: 24 }}
              >
                <Alert
                  message={
                    <Space align="center">
                      <InfoCircleOutlined />
                      <Text strong>使用提示</Text>
                    </Space>
                  }
                  description={
                    <div>
                      <p style={{ color: '#1f2937', fontWeight: 600, textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)', marginBottom: '8px' }}>
                        <strong>📋 图片要求：</strong>支持JPG、PNG格式，文件大小不超过10MB
                      </p>
                      <p style={{ color: '#d4351c', fontWeight: 700, textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)', marginBottom: '8px' }}>
                        <strong>🎯 最佳效果：</strong>推荐上传2-8MB、1920×1080以上的高清图片，图片越大输出越清晰
                      </p>
                      <p style={{ color: '#1f2937', fontWeight: 600, textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)', marginBottom: '8px' }}>
                        <strong>📐 像素处理：</strong>小图智能放大、大图保持原质量、专业刺绣优化
                      </p>
                      <p style={{ color: '#1f2937', fontWeight: 600, textShadow: '0 1px 2px rgba(255, 255, 255, 0.8)', marginBottom: 0 }}>
                        <strong>⏱️ 处理说明：</strong>处理时间约10-60秒，生成的打样图可直接用于织机设备
                      </p>
                    </div>
                  }
                  type="info"
                  showIcon
                  style={{
                    background: 'rgba(255, 255, 255, 0.1)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    backdropFilter: 'blur(10px)'
                  }}
                />
              </Card>
            </Col>
          </Row>

          {/* 页脚信息 */}
          <Card
            className="liquid-glass-card liquid-fade-in liquid-float-delayed"
            style={{ marginTop: 24, textAlign: 'center' }}
          >
            <Divider />
            <Space direction="vertical" size="small">
              <Text type="secondary">
                🧵 蜀锦蜀绣AI打样图生成工具 v1.0.0
              </Text>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                传承千年工艺，融合现代科技 | 专业织机图像处理解决方案
              </Text>
            </Space>
          </Card>
        </div>
      </Content>
    </Layout>
  );
} 