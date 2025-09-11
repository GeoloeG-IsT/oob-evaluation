# Frontend Bundle Optimization Guide

This document provides comprehensive guidance on the bundle optimization and performance monitoring system implemented for the ML Evaluation Platform frontend.

## Overview

The frontend build optimization system includes:

- **Bundle Analysis Tools** - Automated analysis and visualization of bundle sizes
- **Performance Monitoring** - Continuous tracking of build performance and metrics
- **Size Budgets** - Enforced limits on bundle sizes with CI/CD integration
- **Regression Detection** - Automated detection of performance regressions
- **CI/CD Integration** - Seamless integration with GitHub Actions and other CI systems

## Quick Start

### Run Bundle Analysis

```bash
# Basic bundle analysis
npm run build:analyze

# Full analysis with reports
npm run analyze

# Performance monitoring
npm run perf:bundle-stats

# Size limit checking
npm run size-limit
```

### CI/CD Integration

```bash
# Run CI bundle check (used in GitHub Actions)
node scripts/ci-bundle-check.js

# Generate PR comments
node scripts/ci-bundle-check.js --pr-comment

# Run with different output formats
node scripts/ci-bundle-check.js --format=github
node scripts/ci-bundle-check.js --format=json
```

## Bundle Analysis Tools

### 1. Bundle Analyzer Script (`scripts/bundle-analysis.js`)

Comprehensive analysis tool that provides:

- **Bundle Size Analysis** - JavaScript, CSS, and asset sizes
- **Chunk Analysis** - Individual chunk sizes and optimization opportunities
- **Budget Enforcement** - Automatic checking against size budgets
- **Historical Tracking** - Performance metrics over time
- **HTML Reports** - Visual reports with recommendations

**Key Features:**
- Analyzes all static assets in `.next/static/`
- Generates detailed JSON and HTML reports
- Tracks bundle sizes over time
- Provides optimization recommendations
- Integrates with CI/CD pipelines

### 2. Performance Monitor (`scripts/performance-monitor.js`)

Continuous performance monitoring with:

- **Build Time Tracking** - Monitor build performance
- **Bundle Size Monitoring** - Track size changes over time
- **Regression Detection** - Automatically detect performance issues
- **System Metrics** - CPU, memory, and environment tracking
- **Trend Analysis** - Historical performance trends

**Usage:**
```bash
# Run performance check
node scripts/performance-monitor.js

# Configure thresholds
const monitor = new PerformanceMonitor({
  thresholds: {
    buildTime: 60000,     // 60 seconds
    bundleSize: 1048576,  // 1MB
    regressionPercent: 10 // 10% increase threshold
  }
});
```

### 3. CI Bundle Check (`scripts/ci-bundle-check.js`)

CI/CD-focused tool for:

- **Pipeline Integration** - Designed for automated CI/CD workflows
- **PR Comments** - Automatic bundle analysis comments on pull requests
- **Baseline Comparison** - Compare current build with main branch baseline
- **Fail Conditions** - Configurable conditions to fail builds
- **Multiple Output Formats** - Console, JSON, and GitHub Actions formats

## Size Budgets Configuration

### Size Limit Configuration (`.size-limit.js`)

Defines bundle size budgets for different parts of the application:

```javascript
module.exports = [
  {
    name: "Main Bundle (JS)",
    path: ".next/static/chunks/main-*.js",
    limit: "300 KB",
    gzip: true
  },
  {
    name: "Framework Bundle",
    path: ".next/static/chunks/framework-*.js",
    limit: "400 KB",
    gzip: true
  },
  // ... additional configurations
];
```

**Budget Categories:**
- **Main Bundle**: Core application logic (300KB)
- **Framework Bundle**: React/Next.js core (400KB)
- **Webpack Runtime**: Build system runtime (50KB)
- **Commons Bundle**: Shared components (200KB)
- **Page Bundles**: Individual page bundles (150KB)
- **CSS Bundle**: All stylesheets (50KB)
- **Total JavaScript**: Combined JS size (800KB)

### Next.js Configuration (`next.config.ts`)

Advanced webpack and build optimizations:

```typescript
// Bundle splitting configuration
webpack: (config, { dev, isServer }) => {
  if (!dev) {
    config.optimization.splitChunks = {
      chunks: "all",
      cacheGroups: {
        react: {
          test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
          name: "react",
          chunks: "all",
          priority: 20,
        },
        canvas: {
          test: /[\\/]node_modules[\\/](fabric|konva|three|opencv)[\\/]/,
          name: "canvas",
          chunks: "all",
          priority: 15,
        }
      }
    };
  }
  return config;
}
```

**Optimization Features:**
- **Code Splitting** - Automatic chunk splitting by usage patterns
- **Tree Shaking** - Remove unused code
- **Bundle Caching** - Long-term caching with immutable assets
- **Image Optimization** - WebP/AVIF conversion with multiple sizes
- **Compression** - Built-in gzip compression
- **Security Headers** - Content security and caching policies

## Performance Monitoring

### Lighthouse CI Integration (`lighthouserc.js`)

Automated performance testing with Google Lighthouse:

```javascript
module.exports = {
  ci: {
    collect: {
      url: ['http://localhost:3000/', 'http://localhost:3000/upload'],
      numberOfRuns: 3
    },
    assert: {
      assertions: {
        'categories:performance': ['warn', { minScore: 0.8 }],
        'metrics:largest-contentful-paint': ['error', { maxNumericValue: 4000 }],
        'metrics:cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }]
      }
    }
  }
};
```

**Performance Budgets:**
- **Performance Score**: Minimum 80%
- **LCP (Largest Contentful Paint)**: Maximum 4000ms
- **CLS (Cumulative Layout Shift)**: Maximum 0.1
- **TBT (Total Blocking Time)**: Maximum 300ms
- **Bundle Size**: Various limits by resource type

### Historical Data Tracking

Performance data is stored in:
- `performance-data/performance-history.json` - Historical metrics
- `performance-data/latest-metrics.json` - Most recent metrics
- `performance-data/reports/` - Detailed performance reports

## CI/CD Integration

### GitHub Actions Workflow (`.github/workflows/bundle-analysis.yml`)

Automated bundle analysis on:
- **Push to main/develop** - Update baseline metrics
- **Pull Requests** - Compare with baseline and comment on PR
- **Scheduled runs** - Daily performance tracking
- **Manual triggers** - On-demand analysis

**Workflow Steps:**
1. **Build Analysis** - Build app and analyze bundles
2. **Performance Check** - Run performance monitoring
3. **Regression Detection** - Compare with baseline
4. **Report Generation** - Create detailed reports
5. **PR Comments** - Post analysis results to PR
6. **Artifact Upload** - Save reports for review

### PR Comment Example

```markdown
## ✅ Bundle Analysis Report

### 📊 Bundle Metrics
| Metric | Value | Status |
|--------|-------|--------|
| Bundle Size | 456KB | ✅ |
| JavaScript | 380KB | ✅ |
| CSS | 32KB | ✅ |
| Build Time | 12.3s | ✅ |
| Performance Score | 87/100 | ✅ |

### 📈 Change from Baseline
| Metric | Change | Status |
|--------|---------|--------|
| Bundle Size | +2.3% | ✅ |
| Build Time | -1.2% | ✅ |

### 💡 Recommendations
- **MEDIUM**: Consider code splitting for large components
```

## Optimization Strategies

### 1. Code Splitting

Implemented automatically through:
- **Route-based splitting** - Each page is a separate chunk
- **Component-based splitting** - Large components split into separate chunks
- **Library splitting** - Third-party libraries in dedicated chunks
- **Dynamic imports** - Lazy loading for non-critical components

### 2. Tree Shaking

Optimized through:
- **ES Modules** - Use ES module imports for better tree shaking
- **Library aliases** - Map to ES module versions (e.g., lodash → lodash-es)
- **Side effects configuration** - Proper package.json sideEffects flags
- **Webpack optimization** - usedExports and providedExports tracking

### 3. Asset Optimization

- **Image optimization** - Automatic WebP/AVIF conversion
- **Font optimization** - Subset loading and display swap
- **CSS optimization** - Minification and critical CSS extraction
- **Static asset caching** - Long-term caching with content hashing

### 4. Build Performance

- **Webpack build workers** - Parallel processing for faster builds
- **Incremental builds** - Next.js incremental static regeneration
- **Build caching** - Cache Next.js build artifacts
- **Module resolution** - Optimized import paths and aliases

## Monitoring and Alerts

### Performance Regression Detection

Automatically detects regressions in:
- **Build Time** - >20% increase from baseline
- **Bundle Size** - >10% increase from baseline
- **Performance Score** - >10 point decrease
- **Core Web Vitals** - LCP, CLS, TBT threshold violations

### Alert Thresholds

**Critical (Fail Build):**
- Bundle size budget violations
- Performance score < 70
- Build time > 90 seconds
- Core Web Vitals failures

**Warnings (Continue Build):**
- 5-10% performance regressions
- Bundle size approaching limits
- Build time 10-20% increase
- Performance score 70-80

### Baseline Management

- **Automatic Updates** - Baseline updated on main branch pushes
- **Manual Updates** - Update baseline via CI workflow dispatch
- **Historical Tracking** - Keep last 100 build metrics
- **Trend Analysis** - 5-build moving averages for stability

## Troubleshooting

### Common Issues

1. **Bundle Size Violations**
   - Check for accidentally imported large libraries
   - Verify tree shaking is working correctly
   - Consider lazy loading for large components

2. **Build Time Regressions**
   - Enable webpack build workers
   - Check for expensive operations in build process
   - Verify Next.js cache is working

3. **Performance Score Drops**
   - Run Lighthouse locally to debug specific issues
   - Check for render-blocking resources
   - Verify image optimization is working

### Debug Commands

```bash
# Analyze bundle with detailed output
ANALYZE=true npm run build

# Run performance check with verbose logging
DEBUG=true node scripts/performance-monitor.js

# Check size limits with detailed breakdown
npm run size-limit:report

# Generate Lighthouse report locally
npx lhci autorun --upload.target=filesystem
```

## Best Practices

### Development Workflow

1. **Regular Monitoring** - Check bundle size during development
2. **Performance Testing** - Test performance impact of new features
3. **Budget Awareness** - Know current budget usage before adding features
4. **Import Optimization** - Use specific imports instead of barrel exports

### Library Management

1. **Bundle Analysis** - Analyze impact before adding new dependencies
2. **Alternative Evaluation** - Consider lighter alternatives for large libraries
3. **Dynamic Loading** - Load heavy libraries only when needed
4. **Version Management** - Keep dependencies updated for performance improvements

### Code Organization

1. **Component Splitting** - Split large components into smaller, focused ones
2. **Lazy Loading** - Use dynamic imports for route and component-level code splitting
3. **Shared Code** - Extract common functionality to reduce duplication
4. **Asset Management** - Optimize images and other assets before including

## Future Enhancements

### Planned Improvements

1. **ML-Specific Optimizations** - Optimize for Canvas/WebGL libraries
2. **Advanced Caching** - Implement service worker caching strategies  
3. **Bundle Comparison** - Visual diff tools for bundle changes
4. **Performance Budgets** - Per-route performance budgets
5. **Real User Monitoring** - Production performance monitoring

### Integration Opportunities

1. **Performance APIs** - Web Vitals collection in production
2. **Error Tracking** - Bundle analysis integration with error monitoring
3. **A/B Testing** - Performance impact of feature variations
4. **CDN Optimization** - Edge caching and optimization strategies

---

This optimization system ensures the ML Evaluation Platform frontend maintains excellent performance as it scales, with automated monitoring and optimization built into the development workflow.