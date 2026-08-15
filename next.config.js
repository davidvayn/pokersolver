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
  // The continual-resolver API launches the pinned Rust binary and reads its
  // immutable model bundle at runtime. Include both in standalone/server
  // traces instead of relying on Next's static import analysis to discover
  // child-process arguments.
  outputFileTracingIncludes: {
    '/api/practice/resolve': [
      './preflop-solver/target/release/preflop-solver',
      ...resolverArtifactFiles.map(
        (file) => `./preflop-solver/models/practice/${file}`
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
