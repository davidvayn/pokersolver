const fullHandManifests = require('./data/practice/full-hand-manifests.json');

const resolverArtifactFiles = [
  ...new Set(
    fullHandManifests
      .filter(
        (manifest) => manifest?.runtime?.kind === 'rust-continual-resolver-v1'
      )
      .flatMap((manifest) => Object.values(manifest.runtime.artifactFiles ?? {}))
  ),
];
if (
  resolverArtifactFiles.length === 0 ||
  resolverArtifactFiles.some(
    (file) =>
      typeof file !== 'string' ||
      !/^[a-z0-9][a-z0-9.-]*\.json\.gz$/.test(file)
  )
) {
  throw new Error('Continual-resolver manifests contain invalid artifact files');
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  poweredByHeader: false,
  // Keep development output isolated from production builds. Sharing `.next`
  // lets `next build` invalidate files used by a running dev server.
  distDir: process.env.NODE_ENV === 'development' ? '.next-dev' : '.next',
  // Webpack disposes inactive route entries after 60 seconds and buffers
  // only five pages by default. This app has more routes, and disposed App
  // Router entries can return 404 instead of rebuilding.
  onDemandEntries: {
    maxInactiveAge: 24 * 60 * 60 * 1000,
    pagesBufferLength: 16,
  },
  // The continual-resolver API launches the pinned Rust binary and reads its
  // verified binary model caches at runtime. The build verifies each cache
  // against canonical JSON before Next traces this smaller serving payload.
  outputFileTracingIncludes: {
    '/api/practice/resolve': [
      './preflop-solver/target/release/preflop-solver',
      ...resolverArtifactFiles.map(
        (file) => `./preflop-solver/models/practice/${file}.bin`
      ),
    ],
  },
  // Cross-origin isolation is required for SharedArrayBuffer, which the
  // multithreaded WASM postflop solver relies on. These headers make it
  // available both on localhost and on Vercel.
  async headers() {
    return [
      {
        source: '/models/practice/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        source: '/models/solved-spots/:path*',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }],
      },
      {
        source: '/(.*)',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
