import type { NextConfig } from "next";
import withBundleAnalyzer from "@next/bundle-analyzer";

const bundleAnalyzer = withBundleAnalyzer({
  enabled: process.env.ANALYZE === "true",
  openAnalyzer: true,
});

const nextConfig: NextConfig = {
  // Build optimization
  experimental: {
    optimizePackageImports: [
      "react",
      "react-dom",
      "@heroicons/react",
      "lucide-react",
      "framer-motion",
    ],
    webpackBuildWorker: true,
    optimizeCss: true,
    scrollRestoration: true,
  },

  // Performance optimizations
  poweredByHeader: false,
  generateEtags: false,
  compress: true,

  // Image optimization
  images: {
    formats: ["image/webp", "image/avif"],
    minimumCacheTTL: 31536000, // 1 year
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    dangerouslyAllowSVG: true,
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },

  // Bundle splitting and optimization
  webpack: (config, { dev, isServer, webpack }) => {
    // Production optimizations
    if (!dev) {
      // Enable bundle splitting
      config.optimization = {
        ...config.optimization,
        usedExports: true,
        sideEffects: false,
        splitChunks: {
          chunks: "all",
          cacheGroups: {
            default: {
              minChunks: 2,
              priority: -20,
              reuseExistingChunk: true,
            },
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: "vendors",
              priority: -10,
              chunks: "all",
            },
            common: {
              name: "common",
              minChunks: 2,
              chunks: "all",
              enforce: true,
            },
            // Separate large libraries
            react: {
              test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
              name: "react",
              chunks: "all",
              priority: 20,
            },
            // ML/Canvas libraries (when added)
            canvas: {
              test: /[\\/]node_modules[\\/](fabric|konva|three|opencv)[\\/]/,
              name: "canvas",
              chunks: "all",
              priority: 15,
            },
          },
        },
      };

      // Bundle analysis in production builds
      if (process.env.ANALYZE === "true") {
        config.plugins.push(
          new webpack.DefinePlugin({
            "process.env.BUNDLE_ANALYZE": JSON.stringify("true"),
          })
        );
      }
    }

    // Tree shaking optimization
    config.optimization.usedExports = true;
    
    // Custom module resolution for better tree shaking
    config.resolve.alias = {
      ...config.resolve.alias,
      // Optimize lodash imports
      lodash: "lodash-es",
    };

    return config;
  },

  // Headers for caching and security
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "X-XSS-Protection",
            value: "1; mode=block",
          },
        ],
      },
      {
        source: "/static/(.*)",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        source: "/_next/static/(.*)",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
    ];
  },

  // Redirects and rewrites for performance
  async redirects() {
    return [];
  },

  // Static optimization
  trailingSlash: false,
  reactStrictMode: true,
  
  // Output optimization
  output: "standalone",
  
  // TypeScript configuration
  typescript: {
    ignoreBuildErrors: false,
  },

  // ESLint configuration  
  eslint: {
    ignoreDuringBuilds: false,
  },
};

export default bundleAnalyzer(nextConfig);