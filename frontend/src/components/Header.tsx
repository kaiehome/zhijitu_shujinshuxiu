import React from 'react';
import { Layout, Typography, Tag, Space } from 'antd';
// 使用emoji替代图标

const { Header: AntHeader } = Layout;
const { Title, Text } = Typography;

const Header: React.FC = () => {
  return (
    <AntHeader 
      className="liquid-glass-card"
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        backdropFilter: 'blur(30px)',
        background: 'rgba(255, 255, 255, 0.1)',
        border: 'none',
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
      }}
    >
      <div className="liquid-float">
        <Title 
          level={3} 
          style={{ 
            margin: 0, 
            fontSize: '1.5rem',
            fontWeight: 600,
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"',
            letterSpacing: '0.5px',
            textShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
            color: '#1e293b'
          }}
        >
          🧵✨ 蜀锦蜀绣 AI 工具
        </Title>
      </div>
      
      <div 
        className="liquid-float-delayed"
        style={{
          display: 'flex',
          alignItems: 'center',
          height: '100%'
        }}
      >
        <Space size="middle" align="center">
          <Tag 
            className="liquid-tag"
            style={{ 
              fontSize: '14px',
              padding: '6px 16px',
              height: 'auto',
              borderRadius: '20px',
              fontWeight: 600,
              background: 'rgba(255, 255, 255, 0.2)',
              border: '1px solid rgba(255, 255, 255, 0.4)',
              color: '#000000',
              textShadow: '0 1px 3px rgba(255, 255, 255, 0.8)',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            ✨ AI驱动
          </Tag>
          <Tag 
            className="liquid-tag"
            style={{ 
              fontSize: '14px',
              padding: '6px 16px',
              height: 'auto',
              borderRadius: '20px',
              fontWeight: 600,
              background: 'rgba(255, 255, 255, 0.2)',
              border: '1px solid rgba(255, 255, 255, 0.4)',
              color: '#000000',
              textShadow: '0 1px 3px rgba(255, 255, 255, 0.8)',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            🎨 专业版
          </Tag>
          <Tag 
            className="liquid-tag"
            style={{ 
              fontSize: '14px',
              padding: '6px 16px',
              height: 'auto',
              borderRadius: '20px',
              fontWeight: 600,
              background: 'rgba(255, 255, 255, 0.2)',
              border: '1px solid rgba(255, 255, 255, 0.4)',
              color: '#000000',
              textShadow: '0 1px 3px rgba(255, 255, 255, 0.8)',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            🏮 传统工艺
          </Tag>
        </Space>
      </div>
    </AntHeader>
  );
};

export default Header; 