import type { AppProps } from 'next/app';
import { ConfigProvider, theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import 'antd/dist/reset.css';
import '../styles/globals.css';
import { StagewiseToolbar } from '@stagewise/toolbar-react';
import ReactPlugin from '@stagewise-plugins/react';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: '#d4351c', // 中国红主色
          colorSuccess: '#52c41a',
          colorWarning: '#faad14',
          colorError: '#ff4d4f',
          borderRadius: 8,
          fontSize: 14,
        },
        components: {
          Button: {
            primaryShadow: '0 2px 4px rgba(212, 53, 28, 0.2)',
          },
          Upload: {
            colorPrimary: '#d4351c',
          },
        },
      }}
    >
      <Component {...pageProps} />
      {process.env.NODE_ENV === 'development' && (
        <StagewiseToolbar
          config={{
            plugins: [ReactPlugin],
          }}
        />
      )}
    </ConfigProvider>
  );
} 