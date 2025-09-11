#!/usr/bin/env node
/**
 * Performance Monitor for ML Evaluation Platform Frontend
 * 
 * This script provides continuous performance monitoring including:
 * - Build time tracking
 * - Bundle size monitoring  
 * - Performance regression detection
 * - CI/CD integration utilities
 * - Historical performance data
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { performance } = require('perf_hooks');

class PerformanceMonitor {
  constructor(options = {}) {
    this.options = {
      historyLimit: options.historyLimit || 100,
      thresholds: {
        buildTime: options.buildTimeThreshold || 60000, // 60 seconds
        bundleSize: options.bundleSizeThreshold || 1024 * 1024, // 1MB
        regressionPercent: options.regressionPercent || 10, // 10% increase
        ...options.thresholds
      },
      ...options
    };
    
    this.dataDir = path.join(__dirname, '..', 'performance-data');
    this.historyFile = path.join(this.dataDir, 'performance-history.json');
    this.metricsFile = path.join(this.dataDir, 'latest-metrics.json');
    this.reportsDir = path.join(this.dataDir, 'reports');
    
    this.ensureDirectories();
  }

  ensureDirectories() {
    [this.dataDir, this.reportsDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
  }

  async runPerformanceCheck() {
    console.log('🎯 Starting performance monitoring...\n');
    
    try {
      // Collect current metrics
      const metrics = await this.collectMetrics();
      
      // Load historical data
      const history = this.loadHistory();
      
      // Detect regressions
      const regressions = this.detectRegressions(metrics, history);
      
      // Update history
      history.push(metrics);
      this.saveHistory(history);
      
      // Generate report
      const report = await this.generateReport(metrics, regressions, history);
      
      // Save metrics
      this.saveLatestMetrics(metrics);
      
      // Print summary
      this.printSummary(report);
      
      // Check for failures
      if (regressions.critical.length > 0) {
        console.error('❌ Critical performance regressions detected!');
        regressions.critical.forEach(reg => {
          console.error(`   ${reg.metric}: ${reg.change}% increase (${reg.current} vs ${reg.baseline})`);
        });
        process.exit(1);
      }
      
      return report;
      
    } catch (error) {
      console.error('❌ Performance monitoring failed:', error.message);
      process.exit(1);
    }
  }

  async collectMetrics() {
    const startTime = performance.now();
    
    console.log('📊 Collecting performance metrics...');
    
    // Build metrics
    const buildMetrics = await this.collectBuildMetrics();
    
    // Bundle metrics
    const bundleMetrics = await this.collectBundleMetrics();
    
    // System metrics
    const systemMetrics = this.collectSystemMetrics();
    
    // Git information
    const gitInfo = this.collectGitInfo();
    
    const endTime = performance.now();
    const collectionTime = Math.round(endTime - startTime);
    
    return {
      timestamp: new Date().toISOString(),
      collectionTime,
      build: buildMetrics,
      bundle: bundleMetrics,
      system: systemMetrics,
      git: gitInfo,
      score: this.calculatePerformanceScore(buildMetrics, bundleMetrics)
    };
  }

  async collectBuildMetrics() {
    console.log('  🔨 Build metrics...');
    
    const buildStart = performance.now();
    
    try {
      // Clean build
      if (fs.existsSync('.next')) {
        execSync('rm -rf .next', { stdio: 'pipe' });
      }
      
      // Production build
      execSync('npm run build', { 
        stdio: 'pipe',
        env: { ...process.env, NODE_ENV: 'production' }
      });
      
      const buildEnd = performance.now();
      const buildTime = Math.round(buildEnd - buildStart);
      
      // Analyze build output
      const buildAnalysis = this.analyzeBuildOutput();
      
      return {
        buildTime,
        success: true,
        analysis: buildAnalysis,
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      const buildEnd = performance.now();
      const buildTime = Math.round(buildEnd - buildStart);
      
      return {
        buildTime,
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  analyzeBuildOutput() {
    const nextDir = path.join(__dirname, '..', '.next');
    
    if (!fs.existsSync(nextDir)) {
      return { error: 'Build output not found' };
    }
    
    try {
      // Get build manifest
      const buildManifest = path.join(nextDir, 'build-manifest.json');
      let manifest = {};
      
      if (fs.existsSync(buildManifest)) {
        manifest = JSON.parse(fs.readFileSync(buildManifest, 'utf8'));
      }
      
      // Get routes manifest
      const routesManifest = path.join(nextDir, 'routes-manifest.json');
      let routes = {};
      
      if (fs.existsSync(routesManifest)) {
        routes = JSON.parse(fs.readFileSync(routesManifest, 'utf8'));
      }
      
      // Analyze static directory
      const staticDir = path.join(nextDir, 'static');
      const staticAnalysis = this.analyzeStaticDirectory(staticDir);
      
      return {
        manifest,
        routes: routes.staticRoutes || [],
        static: staticAnalysis,
        outputSize: this.getDirectorySize(nextDir)
      };
      
    } catch (error) {
      return { error: `Build analysis failed: ${error.message}` };
    }
  }

  analyzeStaticDirectory(staticDir) {
    if (!fs.existsSync(staticDir)) {
      return { error: 'Static directory not found' };
    }
    
    const analysis = {
      chunks: { count: 0, totalSize: 0, files: [] },
      css: { count: 0, totalSize: 0, files: [] },
      media: { count: 0, totalSize: 0, files: [] }
    };
    
    // Analyze chunks
    const chunksDir = path.join(staticDir, 'chunks');
    if (fs.existsSync(chunksDir)) {
      const chunkFiles = fs.readdirSync(chunksDir).filter(f => f.endsWith('.js'));
      analysis.chunks.count = chunkFiles.length;
      
      chunkFiles.forEach(file => {
        const filePath = path.join(chunksDir, file);
        const size = fs.statSync(filePath).size;
        analysis.chunks.totalSize += size;
        analysis.chunks.files.push({ name: file, size });
      });
    }
    
    // Analyze CSS
    const cssDir = path.join(staticDir, 'css');
    if (fs.existsSync(cssDir)) {
      const cssFiles = fs.readdirSync(cssDir).filter(f => f.endsWith('.css'));
      analysis.css.count = cssFiles.length;
      
      cssFiles.forEach(file => {
        const filePath = path.join(cssDir, file);
        const size = fs.statSync(filePath).size;
        analysis.css.totalSize += size;
        analysis.css.files.push({ name: file, size });
      });
    }
    
    // Analyze media
    const mediaDir = path.join(staticDir, 'media');
    if (fs.existsSync(mediaDir)) {
      const mediaFiles = fs.readdirSync(mediaDir);
      analysis.media.count = mediaFiles.length;
      
      mediaFiles.forEach(file => {
        const filePath = path.join(mediaDir, file);
        const size = fs.statSync(filePath).size;
        analysis.media.totalSize += size;
        analysis.media.files.push({ name: file, size });
      });
    }
    
    return analysis;
  }

  async collectBundleMetrics() {
    console.log('  📦 Bundle metrics...');
    
    const nextDir = path.join(__dirname, '..', '.next');
    const staticDir = path.join(nextDir, 'static');
    
    if (!fs.existsSync(staticDir)) {
      return { error: 'No build output found' };
    }
    
    const metrics = {
      totalSize: this.getDirectorySize(staticDir),
      javascript: this.analyzeBundles(path.join(staticDir, 'chunks'), '.js'),
      css: this.analyzeBundles(path.join(staticDir, 'css'), '.css'),
      assets: this.analyzeBundles(path.join(staticDir, 'media')),
      firstLoadJS: this.calculateFirstLoadJS(staticDir)
    };
    
    // Calculate derived metrics
    metrics.totalJSSize = metrics.javascript.totalSize;
    metrics.totalCSSSize = metrics.css.totalSize;
    metrics.totalAssetsSize = metrics.assets.totalSize;
    metrics.estimatedGzipSize = Math.round((metrics.totalJSSize + metrics.totalCSSSize) * 0.3);
    
    return metrics;
  }

  analyzeBundles(dir, extension = null) {
    if (!fs.existsSync(dir)) {
      return { count: 0, totalSize: 0, files: [], largestFile: null };
    }
    
    let files = fs.readdirSync(dir);
    
    if (extension) {
      files = files.filter(f => f.endsWith(extension));
    }
    
    let totalSize = 0;
    let largestFile = null;
    let largestSize = 0;
    
    const fileData = files.map(file => {
      const filePath = path.join(dir, file);
      const size = fs.statSync(filePath).size;
      totalSize += size;
      
      if (size > largestSize) {
        largestSize = size;
        largestFile = { name: file, size };
      }
      
      return { name: file, size };
    });
    
    return {
      count: files.length,
      totalSize,
      files: fileData.sort((a, b) => b.size - a.size),
      largestFile
    };
  }

  calculateFirstLoadJS(staticDir) {
    const chunksDir = path.join(staticDir, 'chunks');
    if (!fs.existsSync(chunksDir)) {
      return 0;
    }
    
    const criticalChunks = [
      'framework',
      'main',
      'webpack',
      'pages/_app'
    ];
    
    let firstLoadSize = 0;
    const chunkFiles = fs.readdirSync(chunksDir);
    
    criticalChunks.forEach(chunkName => {
      const matchingFiles = chunkFiles.filter(file => 
        file.includes(chunkName) && file.endsWith('.js')
      );
      
      matchingFiles.forEach(file => {
        const filePath = path.join(chunksDir, file);
        firstLoadSize += fs.statSync(filePath).size;
      });
    });
    
    return firstLoadSize;
  }

  collectSystemMetrics() {
    return {
      nodeVersion: process.version,
      platform: process.platform,
      arch: process.arch,
      memory: {
        total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
        used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        external: Math.round(process.memoryUsage().external / 1024 / 1024)
      },
      timestamp: new Date().toISOString()
    };
  }

  collectGitInfo() {
    try {
      return {
        commit: execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim(),
        branch: execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf8' }).trim(),
        author: execSync('git log -1 --pretty=format:"%an"', { encoding: 'utf8' }).trim(),
        message: execSync('git log -1 --pretty=format:"%s"', { encoding: 'utf8' }).trim(),
        timestamp: execSync('git log -1 --pretty=format:"%ai"', { encoding: 'utf8' }).trim()
      };
    } catch (error) {
      return {
        error: 'Git information not available',
        message: error.message
      };
    }
  }

  calculatePerformanceScore(buildMetrics, bundleMetrics) {
    let score = 100;
    
    // Build time impact (max -30 points)
    if (buildMetrics.buildTime > this.options.thresholds.buildTime) {
      const penalty = Math.min(30, (buildMetrics.buildTime / this.options.thresholds.buildTime - 1) * 30);
      score -= penalty;
    }
    
    // Bundle size impact (max -40 points)
    if (bundleMetrics.totalSize > this.options.thresholds.bundleSize) {
      const penalty = Math.min(40, (bundleMetrics.totalSize / this.options.thresholds.bundleSize - 1) * 40);
      score -= penalty;
    }
    
    // First Load JS impact (max -30 points)
    const firstLoadLimit = 600 * 1024; // 600KB
    if (bundleMetrics.firstLoadJS > firstLoadLimit) {
      const penalty = Math.min(30, (bundleMetrics.firstLoadJS / firstLoadLimit - 1) * 30);
      score -= penalty;
    }
    
    return Math.max(0, Math.round(score));
  }

  loadHistory() {
    if (!fs.existsSync(this.historyFile)) {
      return [];
    }
    
    try {
      const history = JSON.parse(fs.readFileSync(this.historyFile, 'utf8'));
      return history.slice(-this.options.historyLimit);
    } catch (error) {
      console.warn('Failed to load performance history:', error.message);
      return [];
    }
  }

  saveHistory(history) {
    // Keep only the latest entries
    const trimmedHistory = history.slice(-this.options.historyLimit);
    fs.writeFileSync(this.historyFile, JSON.stringify(trimmedHistory, null, 2));
  }

  detectRegressions(currentMetrics, history) {
    const regressions = {
      critical: [],
      warnings: [],
      improvements: []
    };
    
    if (history.length < 2) {
      return regressions;
    }
    
    // Use average of last 3 successful builds as baseline
    const recentBuilds = history.filter(h => h.build.success).slice(-3);
    if (recentBuilds.length === 0) {
      return regressions;
    }
    
    const baseline = {
      buildTime: recentBuilds.reduce((sum, b) => sum + b.build.buildTime, 0) / recentBuilds.length,
      totalSize: recentBuilds.reduce((sum, b) => sum + (b.bundle.totalSize || 0), 0) / recentBuilds.length,
      firstLoadJS: recentBuilds.reduce((sum, b) => sum + (b.bundle.firstLoadJS || 0), 0) / recentBuilds.length,
      score: recentBuilds.reduce((sum, b) => sum + b.score, 0) / recentBuilds.length
    };
    
    // Check build time regression
    const buildTimeChange = ((currentMetrics.build.buildTime - baseline.buildTime) / baseline.buildTime) * 100;
    if (buildTimeChange > this.options.thresholds.regressionPercent) {
      regressions.critical.push({
        metric: 'Build Time',
        current: `${currentMetrics.build.buildTime}ms`,
        baseline: `${Math.round(baseline.buildTime)}ms`,
        change: Math.round(buildTimeChange)
      });
    } else if (buildTimeChange > 5) {
      regressions.warnings.push({
        metric: 'Build Time',
        current: `${currentMetrics.build.buildTime}ms`,
        baseline: `${Math.round(baseline.buildTime)}ms`,
        change: Math.round(buildTimeChange)
      });
    }
    
    // Check bundle size regression
    const sizeChange = ((currentMetrics.bundle.totalSize - baseline.totalSize) / baseline.totalSize) * 100;
    if (sizeChange > this.options.thresholds.regressionPercent) {
      regressions.critical.push({
        metric: 'Bundle Size',
        current: `${Math.round(currentMetrics.bundle.totalSize / 1024)}KB`,
        baseline: `${Math.round(baseline.totalSize / 1024)}KB`,
        change: Math.round(sizeChange)
      });
    } else if (sizeChange > 5) {
      regressions.warnings.push({
        metric: 'Bundle Size',
        current: `${Math.round(currentMetrics.bundle.totalSize / 1024)}KB`,
        baseline: `${Math.round(baseline.totalSize / 1024)}KB`,
        change: Math.round(sizeChange)
      });
    }
    
    // Check performance score regression
    const scoreChange = baseline.score - currentMetrics.score;
    if (scoreChange > 10) {
      regressions.critical.push({
        metric: 'Performance Score',
        current: currentMetrics.score,
        baseline: Math.round(baseline.score),
        change: Math.round(scoreChange)
      });
    } else if (scoreChange > 5) {
      regressions.warnings.push({
        metric: 'Performance Score',
        current: currentMetrics.score,
        baseline: Math.round(baseline.score),
        change: Math.round(scoreChange)
      });
    }
    
    // Check for improvements
    if (buildTimeChange < -5) {
      regressions.improvements.push({
        metric: 'Build Time',
        improvement: Math.abs(Math.round(buildTimeChange))
      });
    }
    
    if (sizeChange < -5) {
      regressions.improvements.push({
        metric: 'Bundle Size',
        improvement: Math.abs(Math.round(sizeChange))
      });
    }
    
    return regressions;
  }

  async generateReport(metrics, regressions, history) {
    const report = {
      timestamp: new Date().toISOString(),
      metrics,
      regressions,
      trends: this.analyzeTrends(history),
      summary: this.generateSummary(metrics, regressions),
      recommendations: this.generateRecommendations(metrics, regressions)
    };
    
    // Save report
    const reportFile = path.join(this.reportsDir, `performance-${Date.now()}.json`);
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
    
    return report;
  }

  analyzeTrends(history) {
    if (history.length < 5) {
      return { insufficient_data: true };
    }
    
    const recent = history.slice(-5);
    const buildTimes = recent.map(h => h.build.buildTime);
    const bundleSizes = recent.map(h => h.bundle.totalSize);
    const scores = recent.map(h => h.score);
    
    return {
      buildTime: {
        trend: this.calculateTrend(buildTimes),
        average: buildTimes.reduce((a, b) => a + b, 0) / buildTimes.length,
        min: Math.min(...buildTimes),
        max: Math.max(...buildTimes)
      },
      bundleSize: {
        trend: this.calculateTrend(bundleSizes),
        average: bundleSizes.reduce((a, b) => a + b, 0) / bundleSizes.length,
        min: Math.min(...bundleSizes),
        max: Math.max(...bundleSizes)
      },
      performanceScore: {
        trend: this.calculateTrend(scores),
        average: scores.reduce((a, b) => a + b, 0) / scores.length,
        min: Math.min(...scores),
        max: Math.max(...scores)
      }
    };
  }

  calculateTrend(values) {
    if (values.length < 2) return 'stable';
    
    const first = values[0];
    const last = values[values.length - 1];
    const change = ((last - first) / first) * 100;
    
    if (change > 5) return 'increasing';
    if (change < -5) return 'decreasing';
    return 'stable';
  }

  generateSummary(metrics, regressions) {
    return {
      buildTime: metrics.build.buildTime,
      bundleSize: Math.round(metrics.bundle.totalSize / 1024),
      firstLoadJS: Math.round(metrics.bundle.firstLoadJS / 1024),
      performanceScore: metrics.score,
      criticalIssues: regressions.critical.length,
      warnings: regressions.warnings.length,
      improvements: regressions.improvements.length,
      status: regressions.critical.length > 0 ? 'FAILED' : 
              regressions.warnings.length > 0 ? 'WARNING' : 'PASSED'
    };
  }

  generateRecommendations(metrics, regressions) {
    const recommendations = [];
    
    // Build time recommendations
    if (metrics.build.buildTime > this.options.thresholds.buildTime) {
      recommendations.push({
        type: 'build_time',
        priority: 'high',
        message: 'Consider enabling webpack build worker and optimizing imports',
        action: 'Enable webpackBuildWorker in next.config.js'
      });
    }
    
    // Bundle size recommendations
    if (metrics.bundle.totalSize > this.options.thresholds.bundleSize) {
      recommendations.push({
        type: 'bundle_size',
        priority: 'critical',
        message: 'Bundle size exceeds threshold. Implement code splitting.',
        action: 'Use dynamic imports and lazy loading'
      });
    }
    
    // First Load JS recommendations
    if (metrics.bundle.firstLoadJS > 600 * 1024) {
      recommendations.push({
        type: 'first_load',
        priority: 'high',
        message: 'First Load JS is too large. This impacts initial page load.',
        action: 'Move non-critical code to separate chunks'
      });
    }
    
    // Regression-based recommendations
    regressions.critical.forEach(reg => {
      recommendations.push({
        type: 'regression',
        priority: 'critical',
        message: `${reg.metric} regression detected: ${reg.change}% increase`,
        action: 'Review recent changes and optimize'
      });
    });
    
    return recommendations;
  }

  saveLatestMetrics(metrics) {
    fs.writeFileSync(this.metricsFile, JSON.stringify(metrics, null, 2));
  }

  printSummary(report) {
    const { summary } = report;
    
    console.log('\n🎯 Performance Monitoring Summary');
    console.log('=====================================');
    console.log(`Status: ${summary.status}`);
    console.log(`Build Time: ${summary.buildTime}ms`);
    console.log(`Bundle Size: ${summary.bundleSize}KB`);
    console.log(`First Load JS: ${summary.firstLoadJS}KB`);
    console.log(`Performance Score: ${summary.performanceScore}/100`);
    
    if (summary.criticalIssues > 0) {
      console.log(`\n❌ Critical Issues: ${summary.criticalIssues}`);
      report.regressions.critical.forEach(reg => {
        console.log(`   ${reg.metric}: ${reg.change}% increase`);
      });
    }
    
    if (summary.warnings > 0) {
      console.log(`\n⚠️  Warnings: ${summary.warnings}`);
      report.regressions.warnings.forEach(warn => {
        console.log(`   ${warn.metric}: ${warn.change}% increase`);
      });
    }
    
    if (summary.improvements > 0) {
      console.log(`\n✅ Improvements: ${summary.improvements}`);
      report.regressions.improvements.forEach(imp => {
        console.log(`   ${imp.metric}: ${imp.improvement}% better`);
      });
    }
    
    if (report.recommendations.length > 0) {
      console.log(`\n💡 Recommendations:`);
      report.recommendations.forEach(rec => {
        console.log(`   ${rec.priority.toUpperCase()}: ${rec.message}`);
      });
    }
    
    console.log('=====================================\n');
  }

  getDirectorySize(dirPath) {
    let totalSize = 0;
    
    const calculateSize = (itemPath) => {
      const stats = fs.statSync(itemPath);
      
      if (stats.isDirectory()) {
        const files = fs.readdirSync(itemPath);
        files.forEach(file => {
          calculateSize(path.join(itemPath, file));
        });
      } else {
        totalSize += stats.size;
      }
    };
    
    if (fs.existsSync(dirPath)) {
      calculateSize(dirPath);
    }
    
    return totalSize;
  }
}

// CLI interface
if (require.main === module) {
  const monitor = new PerformanceMonitor();
  
  monitor.runPerformanceCheck().catch(error => {
    console.error('❌ Performance monitoring failed:', error);
    process.exit(1);
  });
}

module.exports = PerformanceMonitor;