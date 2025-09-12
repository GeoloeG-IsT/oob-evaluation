#!/usr/bin/env node
/**
 * CI/CD Bundle Check Script for ML Evaluation Platform Frontend
 * 
 * This script is designed to run in CI/CD pipelines to:
 * - Enforce bundle size budgets
 * - Compare bundle sizes with baseline
 * - Generate CI-friendly reports
 * - Set exit codes for pipeline failures
 * - Generate PR comments with bundle analysis
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const BundleAnalyzer = require('./bundle-analysis');
const PerformanceMonitor = require('./performance-monitor');

class CIBundleCheck {
  constructor(options = {}) {
    this.options = {
      failOnBudgetViolation: options.failOnBudgetViolation !== false,
      failOnRegression: options.failOnRegression !== false,
      generatePRComment: options.generatePRComment || false,
      baselineBranch: options.baselineBranch || 'main',
      outputFormat: options.outputFormat || 'console', // console, json, github
      ...options
    };
    
    this.ciDir = path.join(__dirname, '..', 'ci-reports');
    this.ensureCIDirectory();
  }

  ensureCIDirectory() {
    if (!fs.existsSync(this.ciDir)) {
      fs.mkdirSync(this.ciDir, { recursive: true });
    }
  }

  async runCICheck() {
    console.log('🚀 Running CI Bundle Check...\n');
    
    const startTime = Date.now();
    
    try {
      // Step 1: Run bundle analysis
      console.log('📦 Analyzing current bundle...');
      const analyzer = new BundleAnalyzer();
      const currentReport = await analyzer.analyzeBuild();
      
      // Step 2: Run performance monitoring
      console.log('🎯 Running performance checks...');
      const monitor = new PerformanceMonitor({
        thresholds: {
          buildTime: 90000, // 90 seconds for CI
          bundleSize: 1024 * 1024, // 1MB
          regressionPercent: 15 // More lenient in CI
        }
      });
      const perfReport = await monitor.runPerformanceCheck();
      
      // Step 3: Get baseline comparison (if available)
      console.log('📊 Comparing with baseline...');
      const baselineComparison = await this.getBaselineComparison(currentReport);
      
      // Step 4: Generate CI report
      const ciReport = this.generateCIReport({
        current: currentReport,
        performance: perfReport,
        baseline: baselineComparison,
        startTime,
        endTime: Date.now()
      });
      
      // Step 5: Save reports
      await this.saveReports(ciReport);
      
      // Step 6: Generate PR comment if requested
      if (this.options.generatePRComment) {
        await this.generatePRComment(ciReport);
      }
      
      // Step 7: Output results based on format
      this.outputResults(ciReport);
      
      // Step 8: Determine exit code
      const shouldFail = this.shouldFailBuild(ciReport);
      
      if (shouldFail) {
        console.error('\n❌ CI Bundle Check Failed!');
        console.error('Reasons:');
        shouldFail.reasons.forEach(reason => console.error(`  - ${reason}`));
        process.exit(1);
      } else {
        console.log('\n✅ CI Bundle Check Passed!');
        return ciReport;
      }
      
    } catch (error) {
      console.error('❌ CI Bundle Check failed:', error.message);
      process.exit(1);
    }
  }

  async getBaselineComparison(currentReport) {
    try {
      // Try to get baseline from git (main branch)
      const currentBranch = this.getCurrentBranch();
      
      if (currentBranch === this.options.baselineBranch) {
        console.log('  On baseline branch, skipping comparison');
        return null;
      }
      
      // Check if we can access baseline data
      const baselineData = this.getStoredBaseline();
      
      if (!baselineData) {
        console.log('  No baseline data available');
        return null;
      }
      
      // Compare current with baseline
      const comparison = this.compareWithBaseline(currentReport, baselineData);
      
      return {
        baseline: baselineData,
        comparison,
        available: true
      };
      
    } catch (error) {
      console.warn('  Baseline comparison failed:', error.message);
      return { available: false, error: error.message };
    }
  }

  getStoredBaseline() {
    const baselineFile = path.join(this.ciDir, 'baseline.json');
    
    if (!fs.existsSync(baselineFile)) {
      return null;
    }
    
    try {
      return JSON.parse(fs.readFileSync(baselineFile, 'utf8'));
    } catch (error) {
      console.warn('Failed to read baseline file:', error.message);
      return null;
    }
  }

  compareWithBaseline(current, baseline) {
    const comparison = {
      bundleSize: {
        current: current.bundleStats.totals.js + current.bundleStats.totals.css,
        baseline: baseline.bundleStats?.totals?.js + baseline.bundleStats?.totals?.css || 0,
        change: 0,
        changePercent: 0
      },
      buildTime: {
        current: current.performanceReport.buildTime,
        baseline: baseline.performanceReport?.buildTime || 0,
        change: 0,
        changePercent: 0
      },
      jsSize: {
        current: current.bundleStats.totals.js,
        baseline: baseline.bundleStats?.totals?.js || 0,
        change: 0,
        changePercent: 0
      }
    };
    
    // Calculate changes
    Object.keys(comparison).forEach(metric => {
      const current = comparison[metric].current;
      const baseline = comparison[metric].baseline;
      
      if (baseline > 0) {
        comparison[metric].change = current - baseline;
        comparison[metric].changePercent = ((current - baseline) / baseline) * 100;
      }
    });
    
    return comparison;
  }

  generateCIReport(data) {
    const { current, performance, baseline, startTime, endTime } = data;
    
    return {
      timestamp: new Date().toISOString(),
      duration: endTime - startTime,
      git: this.getGitInfo(),
      ci: this.getCIEnvironmentInfo(),
      
      // Current metrics
      current: {
        bundleSize: Math.round((current.bundleStats.totals.js + current.bundleStats.totals.css) / 1024),
        jsSize: Math.round(current.bundleStats.totals.js / 1024),
        cssSize: Math.round(current.bundleStats.totals.css / 1024),
        buildTime: current.performanceReport.buildTime,
        performanceScore: performance.metrics.score,
        budgetViolations: current.budgetResults.violations.length,
        warnings: current.budgetResults.warnings.length
      },
      
      // Baseline comparison
      baseline: baseline?.available ? {
        bundleSize: Math.round(baseline.comparison.bundleSize.changePercent),
        buildTime: Math.round(baseline.comparison.buildTime.changePercent),
        jsSize: Math.round(baseline.comparison.jsSize.changePercent)
      } : null,
      
      // Results
      results: {
        budgetCheck: current.budgetResults.hasViolations ? 'FAILED' : 'PASSED',
        performanceCheck: performance.summary.status,
        regressionCheck: baseline?.available ? this.assessRegressions(baseline.comparison) : 'SKIPPED'
      },
      
      // Details for debugging
      violations: current.budgetResults.violations,
      recommendations: current.summary.recommendations,
      
      // CI-specific data
      artifacts: {
        bundleReport: path.join(this.ciDir, 'bundle-report.json'),
        performanceReport: path.join(this.ciDir, 'performance-report.json'),
        htmlReport: path.join(this.ciDir, 'bundle-analysis.html')
      }
    };
  }

  assessRegressions(comparison) {
    const thresholds = {
      bundleSize: 10, // 10% increase
      buildTime: 20,  // 20% increase
      jsSize: 10      // 10% increase
    };
    
    const regressions = [];
    
    Object.keys(comparison).forEach(metric => {
      if (thresholds[metric] && comparison[metric].changePercent > thresholds[metric]) {
        regressions.push({
          metric,
          changePercent: Math.round(comparison[metric].changePercent),
          threshold: thresholds[metric]
        });
      }
    });
    
    return regressions.length > 0 ? 'FAILED' : 'PASSED';
  }

  async saveReports(ciReport) {
    // Save main CI report
    const ciReportFile = path.join(this.ciDir, 'ci-report.json');
    fs.writeFileSync(ciReportFile, JSON.stringify(ciReport, null, 2));
    
    // Save simplified report for external tools
    const simplifiedReport = {
      timestamp: ciReport.timestamp,
      status: this.getOverallStatus(ciReport),
      bundleSize: ciReport.current.bundleSize,
      budgetViolations: ciReport.current.budgetViolations,
      performanceScore: ciReport.current.performanceScore,
      buildTime: ciReport.current.buildTime
    };
    
    const simpleReportFile = path.join(this.ciDir, 'simple-report.json');
    fs.writeFileSync(simpleReportFile, JSON.stringify(simplifiedReport, null, 2));
    
    // Update baseline if on main branch
    if (this.getCurrentBranch() === this.options.baselineBranch) {
      await this.updateBaseline(ciReport);
    }
    
    console.log(`📁 CI reports saved to: ${this.ciDir}`);
  }

  async updateBaseline(ciReport) {
    const baselineFile = path.join(this.ciDir, 'baseline.json');
    
    const baseline = {
      timestamp: ciReport.timestamp,
      git: ciReport.git,
      bundleStats: {
        totals: {
          js: ciReport.current.jsSize * 1024,
          css: ciReport.current.cssSize * 1024
        }
      },
      performanceReport: {
        buildTime: ciReport.current.buildTime
      }
    };
    
    fs.writeFileSync(baselineFile, JSON.stringify(baseline, null, 2));
    console.log('📊 Baseline updated');
  }

  async generatePRComment(ciReport) {
    if (!this.isPullRequest()) {
      console.log('Not in PR context, skipping PR comment generation');
      return;
    }
    
    const comment = this.buildPRComment(ciReport);
    
    // Save comment to file for external tools to use
    const commentFile = path.join(this.ciDir, 'pr-comment.md');
    fs.writeFileSync(commentFile, comment);
    
    console.log(`💬 PR comment saved to: ${commentFile}`);
    
    // If GitHub CLI is available, post comment directly
    try {
      execSync(`gh pr comment --body-file "${commentFile}"`, { stdio: 'pipe' });
      console.log('✅ PR comment posted via GitHub CLI');
    } catch (error) {
      console.log('ℹ️  GitHub CLI not available or failed, comment saved to file');
    }
  }

  buildPRComment(ciReport) {
    const status = this.getOverallStatus(ciReport);
    const statusEmoji = status === 'PASSED' ? '✅' : status === 'WARNING' ? '⚠️' : '❌';
    
    let comment = `## ${statusEmoji} Bundle Analysis Report\n\n`;
    
    // Summary table
    comment += '### 📊 Bundle Metrics\n\n';
    comment += '| Metric | Value | Status |\n';
    comment += '|--------|-------|--------|\n';
    comment += `| Bundle Size | ${ciReport.current.bundleSize}KB | ${ciReport.current.bundleSize > 800 ? '❌' : '✅'} |\n`;
    comment += `| JavaScript | ${ciReport.current.jsSize}KB | ${ciReport.current.jsSize > 600 ? '❌' : '✅'} |\n`;
    comment += `| CSS | ${ciReport.current.cssSize}KB | ${ciReport.current.cssSize > 50 ? '❌' : '✅'} |\n`;
    comment += `| Build Time | ${ciReport.current.buildTime}ms | ${ciReport.current.buildTime > 60000 ? '❌' : '✅'} |\n`;
    comment += `| Performance Score | ${ciReport.current.performanceScore}/100 | ${ciReport.current.performanceScore < 80 ? '❌' : '✅'} |\n\n`;
    
    // Baseline comparison
    if (ciReport.baseline) {
      comment += '### 📈 Change from Baseline\n\n';
      comment += '| Metric | Change | Status |\n';
      comment += '|--------|---------|--------|\n';
      
      const bundleChange = ciReport.baseline.bundleSize;
      const buildChange = ciReport.baseline.buildTime;
      
      comment += `| Bundle Size | ${bundleChange > 0 ? '+' : ''}${bundleChange}% | ${Math.abs(bundleChange) > 10 ? '❌' : '✅'} |\n`;
      comment += `| Build Time | ${buildChange > 0 ? '+' : ''}${buildChange}% | ${Math.abs(buildChange) > 20 ? '❌' : '✅'} |\n\n`;
    }
    
    // Budget violations
    if (ciReport.current.budgetViolations > 0) {
      comment += '### ❌ Budget Violations\n\n';
      ciReport.violations.forEach(violation => {
        const overBy = Math.round(violation.difference / 1024);
        comment += `- **${violation.type}**: ${Math.round(violation.actual / 1024)}KB (budget: ${Math.round(violation.budget / 1024)}KB, over by: ${overBy}KB)\n`;
      });
      comment += '\n';
    }
    
    // Recommendations
    if (ciReport.recommendations.length > 0) {
      comment += '### 💡 Recommendations\n\n';
      ciReport.recommendations.forEach(rec => {
        comment += `- **${rec.priority.toUpperCase()}**: ${rec.message}\n`;
      });
      comment += '\n';
    }
    
    comment += '---\n';
    comment += '*Bundle analysis generated by ML Evaluation Platform CI*';
    
    return comment;
  }

  outputResults(ciReport) {
    switch (this.options.outputFormat) {
      case 'json':
        console.log(JSON.stringify(ciReport, null, 2));
        break;
        
      case 'github':
        this.outputGitHubActions(ciReport);
        break;
        
      default:
        this.outputConsole(ciReport);
    }
  }

  outputGitHubActions(ciReport) {
    const status = this.getOverallStatus(ciReport);
    
    // Set GitHub Actions outputs
    console.log(`::set-output name=status::${status}`);
    console.log(`::set-output name=bundle-size::${ciReport.current.bundleSize}`);
    console.log(`::set-output name=build-time::${ciReport.current.buildTime}`);
    console.log(`::set-output name=performance-score::${ciReport.current.performanceScore}`);
    console.log(`::set-output name=violations::${ciReport.current.budgetViolations}`);
    
    // Set annotations for violations
    if (ciReport.current.budgetViolations > 0) {
      ciReport.violations.forEach(violation => {
        console.log(`::warning file=next.config.ts::Bundle size violation: ${violation.type} exceeds budget by ${Math.round(violation.difference / 1024)}KB`);
      });
    }
  }

  outputConsole(ciReport) {
    console.log('\n🎯 CI Bundle Check Results');
    console.log('===========================');
    console.log(`Overall Status: ${this.getOverallStatus(ciReport)}`);
    console.log(`Bundle Size: ${ciReport.current.bundleSize}KB`);
    console.log(`Build Time: ${ciReport.current.buildTime}ms`);
    console.log(`Performance Score: ${ciReport.current.performanceScore}/100`);
    console.log(`Budget Violations: ${ciReport.current.budgetViolations}`);
    
    if (ciReport.baseline) {
      console.log('\nBaseline Comparison:');
      console.log(`Bundle Size Change: ${ciReport.baseline.bundleSize > 0 ? '+' : ''}${ciReport.baseline.bundleSize}%`);
      console.log(`Build Time Change: ${ciReport.baseline.buildTime > 0 ? '+' : ''}${ciReport.baseline.buildTime}%`);
    }
    
    console.log('===========================\n');
  }

  shouldFailBuild(ciReport) {
    const reasons = [];
    
    if (this.options.failOnBudgetViolation && ciReport.current.budgetViolations > 0) {
      reasons.push(`${ciReport.current.budgetViolations} budget violations`);
    }
    
    if (this.options.failOnRegression && ciReport.results.regressionCheck === 'FAILED') {
      reasons.push('Performance regressions detected');
    }
    
    if (ciReport.results.performanceCheck === 'FAILED') {
      reasons.push('Performance check failed');
    }
    
    return reasons.length > 0 ? { shouldFail: true, reasons } : false;
  }

  getOverallStatus(ciReport) {
    if (ciReport.results.budgetCheck === 'FAILED' || 
        ciReport.results.performanceCheck === 'FAILED' ||
        ciReport.results.regressionCheck === 'FAILED') {
      return 'FAILED';
    }
    
    if (ciReport.current.warnings > 0) {
      return 'WARNING';
    }
    
    return 'PASSED';
  }

  getCurrentBranch() {
    try {
      return execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf8' }).trim();
    } catch {
      return process.env.GITHUB_HEAD_REF || process.env.CI_BRANCH || 'unknown';
    }
  }

  isPullRequest() {
    return !!(process.env.GITHUB_EVENT_NAME === 'pull_request' || 
             process.env.CI_PULL_REQUEST ||
             process.env.PULL_REQUEST_NUMBER);
  }

  getGitInfo() {
    try {
      return {
        commit: execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim(),
        branch: this.getCurrentBranch(),
        author: execSync('git log -1 --pretty=format:"%an"', { encoding: 'utf8' }).trim(),
        message: execSync('git log -1 --pretty=format:"%s"', { encoding: 'utf8' }).trim()
      };
    } catch {
      return {
        commit: process.env.GITHUB_SHA || 'unknown',
        branch: this.getCurrentBranch(),
        author: process.env.GITHUB_ACTOR || 'unknown',
        message: 'CI build'
      };
    }
  }

  getCIEnvironmentInfo() {
    return {
      provider: this.detectCIProvider(),
      buildNumber: process.env.GITHUB_RUN_NUMBER || process.env.BUILD_NUMBER || 'unknown',
      buildUrl: process.env.GITHUB_SERVER_URL && process.env.GITHUB_REPOSITORY && process.env.GITHUB_RUN_ID 
        ? `${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`
        : 'unknown',
      isPR: this.isPullRequest()
    };
  }

  detectCIProvider() {
    if (process.env.GITHUB_ACTIONS) return 'github-actions';
    if (process.env.GITLAB_CI) return 'gitlab-ci';
    if (process.env.JENKINS_URL) return 'jenkins';
    if (process.env.CIRCLECI) return 'circle-ci';
    if (process.env.TRAVIS) return 'travis-ci';
    return 'unknown';
  }
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  
  const options = {
    failOnBudgetViolation: !args.includes('--no-fail-budget'),
    failOnRegression: !args.includes('--no-fail-regression'),
    generatePRComment: args.includes('--pr-comment'),
    outputFormat: args.find(arg => arg.startsWith('--format='))?.split('=')[1] || 'console',
    baselineBranch: args.find(arg => arg.startsWith('--baseline='))?.split('=')[1] || 'main'
  };
  
  const checker = new CIBundleCheck(options);
  
  checker.runCICheck().catch(error => {
    console.error('❌ CI Bundle Check failed:', error);
    process.exit(1);
  });
}

module.exports = CIBundleCheck;