/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/outputs/:path*',
        destination: 'http://localhost:8000/outputs/:path*',
      },
    ];
  },
  images: {
    domains: ['localhost'],
  },
  // 禁用严格模式以避免组件重复渲染
  experimental: {
    forceSwcTransforms: true,
  },
};

module.exports = nextConfig; 