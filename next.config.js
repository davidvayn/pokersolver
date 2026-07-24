/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Keep development output isolated from production builds. Sharing `.next`
  // lets `next build` invalidate files used by a running dev server.
  distDir: process.env.NODE_ENV === 'development' ? '.next-dev' : '.next',
  // Next 15.1 disposes inactive route entries after 60 seconds and buffers
  // only five pages by default. This app has more routes, and disposed App
  // Router entries can return 404 instead of rebuilding.
  onDemandEntries: {
    maxInactiveAge: 24 * 60 * 60 * 1000,
    pagesBufferLength: 16,
  },
  // Cross-origin isolation is required for SharedArrayBuffer, which the
  // multithreaded WASM postflop solver relies on. These headers make it
  // available both on localhost and on Vercel.
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
