import { NextPageContext } from 'next';
import React from 'react';

interface ErrorProps {
  statusCode?: number;
}

function Error({ statusCode }: ErrorProps) {
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#fff',
      color: '#222',
      fontFamily: 'sans-serif',
    }}>
      <h1>{statusCode ? `错误码: ${statusCode}` : '发生未知错误'}</h1>
      <p>很抱歉，页面出现了异常。</p>
      <a href="/" style={{ color: '#1677ff', textDecoration: 'underline' }}>返回首页</a>
    </div>
  );
}

Error.getInitialProps = ({ res, err }: NextPageContext) => {
  const statusCode = res ? res.statusCode : err ? err.statusCode : 404;
  return { statusCode };
};

export default Error; 