#!/usr/bin/env node
/**
 * Bundle Analysis Script for ML Evaluation Platform Frontend
 * 
 * This script provides comprehensive bundle analysis including:
 * - Bundle size tracking over time
 * - Performance metrics collection
 * - Build time analysis
 * - Size budget enforcement
 * - Report generation for CI/CD
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { performance } = require('perf_hooks');

class BundleAnalyzer {
  constructor() {
    this.reportsDir = path.join(__dirname, '..', 'reports');
    this.bundleStatsDir = path.join(this.reportsDir, 'bundle-stats');
    this.buildStatsFile = path.join(this.bundleStatsDir, 'build-stats.json');
    this.sizeBudgets = this.loadSizeBudgets();
    
    // Ensure directories exist
    this.ensureDirectoriesExist();
  }

  ensureDirectoriesExist() {
    if (!fs.existsSync(this.reportsDir)) {
      fs.mkdirSync(this.reportsDir, { recursive: true });
    }
    if (!fs.existsSync(this.bundleStatsDir)) {
      fs.mkdirSync(this.bundleStatsDir, { recursive: true });
    }
  }

  loadSizeBudgets() {
    return {
      // Main bundle budgets (in KB)
      'pages/_app': 250,
      'pages/index': 150,
      'chunks/main': 300,
      'chunks/webpack': 50,
      'chunks/framework': 400,
      'chunks/commons': 200,
      
      // Asset budgets
      'css/total': 50,
      'images/total': 500,
      'fonts/total': 100,
      
      // Overall budgets
      'total/js': 800,
      'total/css': 50,
      'total/assets': 1000,
      'gzip/total': 300,
    };
  }

  async analyzeBuild() {
    console.log('🔍 Starting bundle analysis...\n');
    
    const buildStart = performance.now();
    
    try {
      // Step 1: Build with analysis
      console.log('📦 Building application with analysis...');
      const buildResult = await this.buildWithAnalysis();
      
      // Step 2: Analyze bundle sizes
      console.log('📊 Analyzing bundle sizes...');
      const bundleStats = await this.analyzeBundleSizes();
      
      // Step 3: Check size budgets
      console.log('💰 Checking size budgets...');
      const budgetResults = this.checkSizeBudgets(bundleStats);
      
      // Step 4: Generate performance report
      console.log('📈 Generating performance report...');
      const performanceReport = await this.generatePerformanceReport(bundleStats, buildResult);
      
      // Step 5: Save historical data
      console.log('💾 Saving historical data...');
      await this.saveHistoricalData(bundleStats, performanceReport);
      
      // Step 6: Generate final report
      const buildEnd = performance.now();
      const totalTime = buildEnd - buildStart;
      
      const finalReport = {
        timestamp: new Date().toISOString(),
        buildTime: Math.round(totalTime),
        bundleStats,
        budgetResults,
        performanceReport,
        summary: this.generateSummary(bundleStats, budgetResults, totalTime)
      };
      
      await this.generateFinalReport(finalReport);
      
      console.log('✅ Bundle analysis complete!\n');
      this.printSummary(finalReport.summary);
      
      // Exit with error code if budgets are exceeded
      if (budgetResults.hasViolations) {
        console.error('❌ Size budget violations detected!');
        process.exit(1);
      }
      
      return finalReport;
      
    } catch (error) {
      console.error('❌ Bundle analysis failed:', error.message);
      process.exit(1);
    }
  }

  async buildWithAnalysis() {
    const buildStart = performance.now();
    
    try {
      // Build with production optimizations
      execSync('npm run build', {
        stdio: 'pipe',
        env: { ...process.env, NODE_ENV: 'production' }
      });
      
      const buildEnd = performance.now();
      const buildTime = buildEnd - buildStart;
      
      return {
        success: true,
        buildTime: Math.round(buildTime),
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      throw new Error(`Build failed: ${error.message}`);
    }
  }

  async analyzeBundleSizes() {
    const nextDir = path.join(__dirname, '..', '.next');
    const staticDir = path.join(nextDir, 'static');
    
    if (!fs.existsSync(staticDir)) {
      throw new Error('Build output not found. Please run build first.');
    }
    
    const stats = {
      javascript: this.analyzeJavaScriptBundles(staticDir),
      css: this.analyzeCSSBundles(staticDir),
      assets: this.analyzeAssets(staticDir),
      pages: this.analyzePages(nextDir),
      chunks: this.analyzeChunks(staticDir)
    };
    
    // Calculate totals
    stats.totals = {
      js: stats.javascript.totalSize,
      css: stats.css.totalSize,
      assets: stats.assets.totalSize,
      gzipped: this.calculateGzippedSize(stats)
    };
    
    return stats;
  }

  analyzeJavaScriptBundles(staticDir) {
    const chunksDir = path.join(staticDir, 'chunks');
    const files = this.getAllFiles(chunksDir, '.js');
    
    let totalSize = 0;
    const bundles = files.map(file => {
      const filePath = path.join(chunksDir, file);
      const size = fs.statSync(filePath).size;
      totalSize += size;
      
      return {
        name: file,
        size: size,
        sizeKB: Math.round(size / 1024),
        path: filePath
      };
    });
    
    // Sort by size descending
    bundles.sort((a, b) => b.size - a.size);
    
    return {
      files: bundles,
      totalSize: totalSize,
      totalSizeKB: Math.round(totalSize / 1024),
      count: bundles.length
    };
  }

  analyzeCSSBundles(staticDir) {
    const cssDir = path.join(staticDir, 'css');
    if (!fs.existsSync(cssDir)) {
      return { files: [], totalSize: 0, totalSizeKB: 0, count: 0 };
    }
    
    const files = this.getAllFiles(cssDir, '.css');
    
    let totalSize = 0;
    const bundles = files.map(file => {
      const filePath = path.join(cssDir, file);
      const size = fs.statSync(filePath).size;
      totalSize += size;
      
      return {
        name: file,
        size: size,
        sizeKB: Math.round(size / 1024),
        path: filePath
      };
    });
    
    bundles.sort((a, b) => b.size - a.size);
    
    return {
      files: bundles,
      totalSize: totalSize,
      totalSizeKB: Math.round(totalSize / 1024),
      count: bundles.length
    };
  }

  analyzeAssets(staticDir) {
    const mediaDir = path.join(staticDir, 'media');
    if (!fs.existsSync(mediaDir)) {
      return { files: [], totalSize: 0, totalSizeKB: 0, count: 0 };
    }
    
    const files = this.getAllFiles(mediaDir);
    
    let totalSize = 0;
    const assets = files.map(file => {
      const filePath = path.join(mediaDir, file);
      const size = fs.statSync(filePath).size;
      totalSize += size;
      
      return {
        name: file,
        size: size,
        sizeKB: Math.round(size / 1024),
        type: path.extname(file),
        path: filePath
      };
    });
    
    assets.sort((a, b) => b.size - a.size);
    
    return {
      files: assets,
      totalSize: totalSize,
      totalSizeKB: Math.round(totalSize / 1024),
      count: assets.length
    };
  }

  analyzePages(nextDir) {
    const pagesManifest = path.join(nextDir, 'server', 'pages-manifest.json');
    if (!fs.existsSync(pagesManifest)) {
      return { pages: [], count: 0 };
    }
    
    const manifest = JSON.parse(fs.readFileSync(pagesManifest, 'utf8'));
    const pages = Object.entries(manifest).map(([route, file]) => ({
      route,
      file,
      exists: fs.existsSync(path.join(nextDir, 'server', file))
    }));
    
    return {
      pages,
      count: pages.length
    };
  }

  analyzeChunks(staticDir) {
    const chunksDir = path.join(staticDir, 'chunks');
    const buildManifest = path.join(staticDir, 'chunks', '_buildManifest.js');
    
    if (!fs.existsSync(buildManifest)) {
      return { chunks: [], count: 0 };
    }
    
    const files = this.getAllFiles(chunksDir, '.js');
    const chunks = files.map(file => {
      const filePath = path.join(chunksDir, file);
      const size = fs.statSync(filePath).size;
      
      return {
        name: file,
        size: size,
        sizeKB: Math.round(size / 1024),
        isFramework: file.includes('framework'),
        isMain: file.includes('main'),
        isWebpack: file.includes('webpack'),
        isVendor: file.includes('vendor')
      };
    });
    
    return {
      chunks: chunks.sort((a, b) => b.size - a.size),
      count: chunks.length
    };
  }

  checkSizeBudgets(bundleStats) {
    const violations = [];
    const warnings = [];
    
    // Check JavaScript budget
    if (bundleStats.totals.js > this.sizeBudgets['total/js'] * 1024) {
      violations.push({
        type: 'JavaScript Total',
        actual: bundleStats.totals.js,
        budget: this.sizeBudgets['total/js'] * 1024,
        difference: bundleStats.totals.js - (this.sizeBudgets['total/js'] * 1024)
      });
    }
    
    // Check CSS budget
    if (bundleStats.totals.css > this.sizeBudgets['total/css'] * 1024) {
      violations.push({
        type: 'CSS Total',
        actual: bundleStats.totals.css,
        budget: this.sizeBudgets['total/css'] * 1024,
        difference: bundleStats.totals.css - (this.sizeBudgets['total/css'] * 1024)
      });
    }
    
    // Check individual chunk budgets
    bundleStats.chunks.chunks.forEach(chunk => {
      const chunkBudgetKey = this.getChunkBudgetKey(chunk.name);
      if (chunkBudgetKey && this.sizeBudgets[chunkBudgetKey]) {
        const budgetBytes = this.sizeBudgets[chunkBudgetKey] * 1024;
        if (chunk.size > budgetBytes) {
          violations.push({
            type: `Chunk: ${chunk.name}`,
            actual: chunk.size,
            budget: budgetBytes,
            difference: chunk.size - budgetBytes
          });
        }
      }
    });
    
    return {
      hasViolations: violations.length > 0,
      hasWarnings: warnings.length > 0,
      violations,
      warnings,
      summary: {
        totalViolations: violations.length,
        totalWarnings: warnings.length,
        overBudgetBy: violations.reduce((sum, v) => sum + v.difference, 0)
      }
    };
  }

  getChunkBudgetKey(chunkName) {
    if (chunkName.includes('framework')) return 'chunks/framework';
    if (chunkName.includes('main')) return 'chunks/main';
    if (chunkName.includes('webpack')) return 'chunks/webpack';
    if (chunkName.includes('commons')) return 'chunks/commons';
    return null;
  }

  calculateGzippedSize(stats) {
    // Estimate gzipped size (typically 70% compression for JS/CSS)
    const jsGzipped = Math.round(stats.totals.js * 0.3);
    const cssGzipped = Math.round(stats.totals.css * 0.3);
    return jsGzipped + cssGzipped;
  }

  async generatePerformanceReport(bundleStats, buildResult) {
    return {
      buildTime: buildResult.buildTime,
      bundleCount: {
        javascript: bundleStats.javascript.count,
        css: bundleStats.css.count,
        assets: bundleStats.assets.count,
        chunks: bundleStats.chunks.count,
        pages: bundleStats.pages.count
      },
      sizeMetrics: {
        totalSize: bundleStats.totals.js + bundleStats.totals.css + bundleStats.totals.assets,
        jsSize: bundleStats.totals.js,
        cssSize: bundleStats.totals.css,
        assetsSize: bundleStats.totals.assets,
        estimatedGzipSize: bundleStats.totals.gzipped
      },
      largestBundles: {
        javascript: bundleStats.javascript.files.slice(0, 5),
        css: bundleStats.css.files.slice(0, 3),
        assets: bundleStats.assets.files.slice(0, 3)
      }
    };
  }

  async saveHistoricalData(bundleStats, performanceReport) {
    const historyFile = path.join(this.bundleStatsDir, 'history.json');
    let history = [];
    
    if (fs.existsSync(historyFile)) {
      history = JSON.parse(fs.readFileSync(historyFile, 'utf8'));
    }
    
    const entry = {
      timestamp: new Date().toISOString(),
      commit: this.getGitCommit(),
      branch: this.getGitBranch(),
      buildTime: performanceReport.buildTime,
      sizes: {
        js: bundleStats.totals.js,
        css: bundleStats.totals.css,
        assets: bundleStats.totals.assets,
        total: bundleStats.totals.js + bundleStats.totals.css + bundleStats.totals.assets,
        gzipped: bundleStats.totals.gzipped
      }
    };
    
    history.push(entry);
    
    // Keep last 50 entries
    if (history.length > 50) {
      history = history.slice(-50);
    }
    
    fs.writeFileSync(historyFile, JSON.stringify(history, null, 2));
  }

  generateSummary(bundleStats, budgetResults, buildTime) {
    return {
      buildTime: Math.round(buildTime),
      totalSize: bundleStats.totals.js + bundleStats.totals.css + bundleStats.totals.assets,
      totalSizeKB: Math.round((bundleStats.totals.js + bundleStats.totals.css + bundleStats.totals.assets) / 1024),
      gzippedSizeKB: Math.round(bundleStats.totals.gzipped / 1024),
      bundleCount: bundleStats.javascript.count + bundleStats.css.count,
      budgetStatus: budgetResults.hasViolations ? 'EXCEEDED' : 'OK',
      violations: budgetResults.violations.length,
      recommendations: this.generateRecommendations(bundleStats, budgetResults)
    };
  }

  generateRecommendations(bundleStats, budgetResults) {
    const recommendations = [];
    
    // Large bundle recommendations
    const largeJSBundles = bundleStats.javascript.files.filter(f => f.sizeKB > 100);
    if (largeJSBundles.length > 0) {
      recommendations.push({
        type: 'optimization',
        priority: 'high',
        message: `Consider code splitting for ${largeJSBundles.length} large JS bundles (>100KB)`,
        bundles: largeJSBundles.map(b => b.name)
      });
    }
    
    // Budget violations
    if (budgetResults.hasViolations) {
      recommendations.push({
        type: 'budget',
        priority: 'critical',
        message: `${budgetResults.violations.length} size budget violations need immediate attention`,
        violations: budgetResults.violations
      });
    }
    
    // Tree shaking opportunities
    if (bundleStats.javascript.files.some(f => f.name.includes('node_modules'))) {
      recommendations.push({
        type: 'tree-shaking',
        priority: 'medium',
        message: 'Review imports for better tree shaking opportunities'
      });
    }
    
    return recommendations;
  }

  async generateFinalReport(report) {
    const reportFile = path.join(this.bundleStatsDir, `report-${Date.now()}.json`);
    const htmlReport = path.join(this.bundleStatsDir, `report-${Date.now()}.html`);
    
    // Save JSON report
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
    
    // Generate HTML report
    const html = this.generateHTMLReport(report);
    fs.writeFileSync(htmlReport, html);
    
    // Update latest report links
    const latestJson = path.join(this.bundleStatsDir, 'latest.json');
    const latestHtml = path.join(this.bundleStatsDir, 'latest.html');
    
    fs.writeFileSync(latestJson, JSON.stringify(report, null, 2));
    fs.writeFileSync(latestHtml, html);
    
    console.log(`📊 Reports saved:`);
    console.log(`   JSON: ${reportFile}`);
    console.log(`   HTML: ${htmlReport}`);
  }

  generateHTMLReport(report) {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bundle Analysis Report - ${new Date(report.timestamp).toLocaleDateString()}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { border-bottom: 2px solid #e1e5e9; padding-bottom: 20px; margin-bottom: 30px; }
        .status { padding: 10px 15px; border-radius: 4px; margin: 10px 0; }
        .status.ok { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .metric { display: inline-block; margin: 15px 20px 15px 0; padding: 15px; background: #f8f9fa; border-radius: 4px; min-width: 120px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 12px; color: #6c757d; text-transform: uppercase; }
        .section { margin: 30px 0; }
        .section h3 { color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
        th { background: #f8f9fa; font-weight: 600; }
        .size-bar { height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; position: relative; }
        .size-fill { height: 100%; background: linear-gradient(90deg, #007bff, #0056b3); border-radius: 10px; }
        .recommendations { background: #e7f3ff; padding: 20px; border-radius: 4px; border-left: 4px solid #007bff; }
        .violation { background: #f8d7da; padding: 10px; margin: 5px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Bundle Analysis Report</h1>
            <p><strong>Generated:</strong> ${new Date(report.timestamp).toLocaleString()}</p>
            <p><strong>Build Time:</strong> ${report.buildTime}ms</p>
            <div class="status ${report.budgetResults.hasViolations ? 'error' : 'ok'}">
                <strong>Budget Status:</strong> ${report.summary.budgetStatus}
                ${report.budgetResults.hasViolations ? `(${report.budgetResults.violations.length} violations)` : ''}
            </div>
        </div>

        <div class="section">
            <h2>Summary Metrics</h2>
            <div class="metric">
                <div class="metric-value">${report.summary.totalSizeKB}KB</div>
                <div class="metric-label">Total Size</div>
            </div>
            <div class="metric">
                <div class="metric-value">${report.summary.gzippedSizeKB}KB</div>
                <div class="metric-label">Gzipped</div>
            </div>
            <div class="metric">
                <div class="metric-value">${report.summary.bundleCount}</div>
                <div class="metric-label">Bundles</div>
            </div>
            <div class="metric">
                <div class="metric-value">${report.summary.buildTime}ms</div>
                <div class="metric-label">Build Time</div>
            </div>
        </div>

        ${report.budgetResults.hasViolations ? `
        <div class="section">
            <h3>❌ Budget Violations</h3>
            ${report.budgetResults.violations.map(v => `
                <div class="violation">
                    <strong>${v.type}:</strong> ${Math.round(v.actual / 1024)}KB 
                    (budget: ${Math.round(v.budget / 1024)}KB, 
                    over by: ${Math.round(v.difference / 1024)}KB)
                </div>
            `).join('')}
        </div>
        ` : ''}

        <div class="section">
            <h3>📦 JavaScript Bundles</h3>
            <table>
                <thead>
                    <tr><th>Bundle</th><th>Size</th><th>Visual</th></tr>
                </thead>
                <tbody>
                    ${report.bundleStats.javascript.files.slice(0, 10).map(bundle => {
                      const maxSize = Math.max(...report.bundleStats.javascript.files.map(b => b.size));
                      const percentage = (bundle.size / maxSize) * 100;
                      return `
                        <tr>
                            <td>${bundle.name}</td>
                            <td>${bundle.sizeKB}KB</td>
                            <td>
                                <div class="size-bar">
                                    <div class="size-fill" style="width: ${percentage}%"></div>
                                </div>
                            </td>
                        </tr>
                      `;
                    }).join('')}
                </tbody>
            </table>
        </div>

        ${report.summary.recommendations.length > 0 ? `
        <div class="section">
            <h3>💡 Recommendations</h3>
            <div class="recommendations">
                ${report.summary.recommendations.map(rec => `
                    <div style="margin: 10px 0;">
                        <strong>${rec.priority.toUpperCase()}:</strong> ${rec.message}
                    </div>
                `).join('')}
            </div>
        </div>
        ` : ''}

        <div class="section">
            <h3>📊 Bundle Breakdown</h3>
            <table>
                <tr><td>JavaScript</td><td>${Math.round(report.bundleStats.totals.js / 1024)}KB</td></tr>
                <tr><td>CSS</td><td>${Math.round(report.bundleStats.totals.css / 1024)}KB</td></tr>
                <tr><td>Assets</td><td>${Math.round(report.bundleStats.totals.assets / 1024)}KB</td></tr>
                <tr><td><strong>Total</strong></td><td><strong>${report.summary.totalSizeKB}KB</strong></td></tr>
                <tr><td><strong>Estimated Gzipped</strong></td><td><strong>${report.summary.gzippedSizeKB}KB</strong></td></tr>
            </table>
        </div>
    </div>
</body>
</html>`;
  }

  printSummary(summary) {
    console.log('📊 Bundle Analysis Summary');
    console.log('================================');
    console.log(`Build Time: ${summary.buildTime}ms`);
    console.log(`Total Size: ${summary.totalSizeKB}KB`);
    console.log(`Gzipped: ${summary.gzippedSizeKB}KB`);
    console.log(`Bundles: ${summary.bundleCount}`);
    console.log(`Budget Status: ${summary.budgetStatus}`);
    
    if (summary.violations > 0) {
      console.log(`\n⚠️  ${summary.violations} budget violations found!`);
    }
    
    if (summary.recommendations.length > 0) {
      console.log(`\n💡 ${summary.recommendations.length} recommendations:`);
      summary.recommendations.forEach(rec => {
        console.log(`   ${rec.priority.toUpperCase()}: ${rec.message}`);
      });
    }
    
    console.log('================================\n');
  }

  getAllFiles(dir, extension = null) {
    if (!fs.existsSync(dir)) return [];
    
    return fs.readdirSync(dir)
      .filter(file => {
        const isFile = fs.statSync(path.join(dir, file)).isFile();
        if (!isFile) return false;
        if (extension) return file.endsWith(extension);
        return true;
      });
  }

  getGitCommit() {
    try {
      return execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
    } catch {
      return 'unknown';
    }
  }

  getGitBranch() {
    try {
      return execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf8' }).trim();
    } catch {
      return 'unknown';
    }
  }
}

// CLI interface
if (require.main === module) {
  const analyzer = new BundleAnalyzer();
  
  analyzer.analyzeBuild().catch(error => {
    console.error('❌ Bundle analysis failed:', error);
    process.exit(1);
  });
}

module.exports = BundleAnalyzer;