import React, { useState } from 'react';
import {
  Upload,
  Button,
  message,
  Space,
  Typography,
  Card,
  Progress,
  Image,
} from 'antd';
import InboxOutlined from '@ant-design/icons/InboxOutlined';
import UploadOutlined from '@ant-design/icons/UploadOutlined';
import CheckCircleOutlined from '@ant-design/icons/CheckCircleOutlined';
import FileImageOutlined from '@ant-design/icons/FileImageOutlined';
import type { UploadProps, UploadFile } from 'antd';
import axios from 'axios';

const { Dragger } = Upload;
const { Text, Paragraph } = Typography;

interface UploadSectionProps {
  onUploadSuccess: (file: any) => void;
  onError?: (errorMessage: string) => void;
  disabled?: boolean;
  uploadedFile: any;
}

const UploadSection: React.FC<UploadSectionProps> = ({ onUploadSuccess, onError, disabled = false, uploadedFile }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // 格式化上传时间
  const formatUploadTime = (timeString: string): string => {
    if (!timeString) return '刚刚';
    
    try {
      const date = new Date(timeString);
      
      // 检查日期是否有效
      if (isNaN(date.getTime())) {
        return '刚刚';
      }
      
      const now = new Date();
      const diff = now.getTime() - date.getTime();
      const minutes = Math.floor(diff / (1000 * 60));
      const hours = Math.floor(diff / (1000 * 60 * 60));
      const days = Math.floor(diff / (1000 * 60 * 60 * 24));
      
      // 相对时间显示
      if (minutes < 1) {
        return '刚刚';
      } else if (minutes < 60) {
        return `${minutes}分钟前`;
      } else if (hours < 24) {
        return `${hours}小时前`;
      } else if (days < 7) {
        return `${days}天前`;
      } else {
        // 超过一周显示具体时间
        return date.toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          hour12: false
        });
      }
    } catch (error) {
      console.warn('时间格式化失败:', timeString, error);
      return '时间未知';
    }
  };

  const customUpload = async (options: any) => {
    const { file, onSuccess, onError, onProgress } = options;
    
    try {
      setUploading(true);
      setUploadProgress(0);
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
          setUploadProgress(percent);
          onProgress({ percent });
        },
      });
      
      if (response.status === 200) {
        message.success('文件上传成功！');
        
        // 将原始文件对象添加到响应数据中
        const enrichedData = {
          ...response.data,
          file: file  // 保留原始文件对象
        };
        
        onSuccess(enrichedData);
        onUploadSuccess(enrichedData);
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || '上传失败，请重试';
      message.error(errorMessage);
      onError?.(errorMessage);
      onError(error);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const beforeUpload = (file: File) => {
    // 检查文件类型
    const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/jpg';
    if (!isJpgOrPng) {
      message.error({
        content: `不支持的文件格式：${file.type || '未知'}。仅支持 JPG、PNG 格式的图片文件！`,
        duration: 4,
      });
      return false;
    }
    
    // 检查文件大小
    const isLt10M = file.size / 1024 / 1024 < 10;
    if (!isLt10M) {
      const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
      message.error({
        content: `文件过大：${fileSizeMB}MB。图片大小不能超过 10MB！`,
        duration: 4,
      });
      return false;
    }
    
    return true;
  };

  const props: UploadProps = {
    name: 'file',
    multiple: false,
    customRequest: customUpload,
    beforeUpload,
    // 限制文件选择器只显示支持的图片格式
    accept: '.jpg,.jpeg,.png,image/jpeg,image/png',
    onChange(info) {
      const { status } = info.file;
      if (status === 'done') {
        message.success(`${info.file.name} 文件上传成功`);
      } else if (status === 'error') {
        message.error(`${info.file.name} 文件上传失败`);
      }
    },
    showUploadList: false,
  };

  if (uploadedFile) {
    return (
      <Card
        style={{
          background: 'rgba(82, 196, 26, 0.1)',
          border: '2px dashed #52c41a',
          borderRadius: 8,
        }}
      >
        <Space direction="vertical" style={{ width: '100%' }} align="center">
          <CheckCircleOutlined style={{ fontSize: 48, color: '#52c41a' }} />
          <Text strong style={{ fontSize: 16 }}>上传成功</Text>
          <Space direction="vertical" style={{ textAlign: 'center' }}>
            <Text><FileImageOutlined /> {uploadedFile.filename}</Text>
            <Text type="secondary">
              大小: {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
            </Text>
            <Text type="secondary">
              上传时间: {formatUploadTime(uploadedFile.uploadTime)}
            </Text>
          </Space>
        </Space>
      </Card>
    );
  }

  return (
    <Space direction="vertical" style={{ width: '100%' }}>
      <Dragger {...props} disabled={uploading || disabled}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined style={{ color: '#d4351c' }} />
        </p>
        <p className="ant-upload-text">点击或拖拽图片到此区域上传</p>
        <p className="ant-upload-hint">
          <strong>📋 图片要求：</strong>仅支持 JPG、PNG 格式，文件大小不超过 10MB
          <br />
          <strong>🎯 最佳效果：</strong>推荐上传 <span style={{color: '#d4351c', fontWeight: 'bold'}}>2-8MB、1920×1080以上</span> 的高清图片
          <br />
          <strong>📐 像素说明：</strong>图片越大输出越清晰 - 小图会智能放大，大图保持原质量
          <br />
          💡 传统图案、工艺品照片效果最佳
        </p>
      </Dragger>
      
      {uploading && (
        <Progress
          percent={uploadProgress}
          status="active"
          strokeColor="#d4351c"
          style={{ marginTop: 16 }}
        />
      )}
      
      <div style={{ textAlign: 'center', marginTop: 16 }}>
        <Paragraph type="secondary" style={{ marginBottom: 8 }}>
          <strong>📊 像素输出说明：</strong>
        </Paragraph>
        <Paragraph type="secondary" style={{ fontSize: '12px', lineHeight: '1.6' }}>
          • <strong>小图片(&lt;1MB)：</strong>自动放大到适合刺绣的尺寸<br/>
          • <strong>推荐范围(2-8MB)：</strong>保持原尺寸，获得最佳刺绣效果<br/>
          • <strong>大图片(&gt;8MB)：</strong>保持高清质量，支持精细刺绣工艺
        </Paragraph>
      </div>
    </Space>
  );
};

export default UploadSection; 