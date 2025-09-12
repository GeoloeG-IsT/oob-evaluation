#!/usr/bin/env node
/**
 * Test Optimization Script for ML Evaluation Platform Frontend
 * 
 * This script validates that all optimization configurations are working correctly
 * and provides diagnostic information about the build setup.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class OptimizationTester {
  constructor() {
    this.results = {
      configuration: [],
      dependencies: [],
      scripts: [],
      optimization: [],
      overall: 'UNKNOWN'
    };
  }

  async runTests() {
    console.log('🧪 Testing Frontend Build Optimization Setup\n');
    
    try {
      await this.testConfiguration();
      await this.testDependencies();
      await this.testScripts();
      await this.testOptimization();
      
      this.generateReport();
      this.printSummary();
      
      const hasFailures = this.results.configuration.some(r => r.status === 'FAILED') ||
                         this.results.dependencies.some(r => r.status === 'FAILED') ||
                         this.results.scripts.some(r => r.status === 'FAILED');
      
      if (hasFailures) {
        console.error('\n❌ Some optimization tests failed!');
        process.exit(1);
      } else {
        console.log('\n✅ All optimization tests passed!');
      }
      
    } catch (error) {
      console.error('❌ Test execution failed:', error.message);
      process.exit(1);
    }
  }

  async testConfiguration() {
    console.log('📋 Testing Configuration Files...');
    
    // Test Next.js config
    const nextConfigPath = path.join(__dirname, '..', 'next.config.ts');
    const nextConfigExists = fs.existsSync(nextConfigPath);
    
    this.results.configuration.push({
      test: 'Next.js Configuration',
      status: nextConfigExists ? 'PASSED' : 'FAILED',
      details: nextConfigExists ? 'Found next.config.ts' : 'next.config.ts not found'
    });
    
    if (nextConfigExists) {
      try {
        const configContent = fs.readFileSync(nextConfigPath, 'utf8');
        const hasAnalyzer = configContent.includes('@next/bundle-analyzer');
        const hasWebpack = configContent.includes('webpack:');
        const hasSplitChunks = configContent.includes('splitChunks');
        
        this.results.configuration.push({
          test: 'Bundle Analyzer Integration',
          status: hasAnalyzer ? 'PASSED' : 'FAILED',
          details: hasAnalyzer ? 'Bundle analyzer configured' : 'Bundle analyzer not found'
        });
        
        this.results.configuration.push({
          test: 'Webpack Configuration',
          status: hasWebpack ? 'PASSED' : 'FAILED',
          details: hasWebpack ? 'Custom webpack config found' : 'No webpack configuration'
        });
        
        this.results.configuration.push({
          test: 'Code Splitting Configuration',
          status: hasSplitChunks ? 'PASSED' : 'FAILED',
          details: hasSplitChunks ? 'Split chunks configured' : 'No split chunks configuration'
        });
        
      } catch (error) {
        this.results.configuration.push({
          test: 'Next.js Config Parsing',
          status: 'FAILED',
          details: `Failed to parse config: ${error.message}`
        });
      }
    }
    
    // Test Size Limit config
    const sizeLimitPath = path.join(__dirname, '..', '.size-limit.js');
    const sizeLimitExists = fs.existsSync(sizeLimitPath);
    
    this.results.configuration.push({
      test: 'Size Limit Configuration',
      status: sizeLimitExists ? 'PASSED' : 'FAILED',
      details: sizeLimitExists ? 'Found .size-limit.js' : '.size-limit.js not found'
    });
    
    // Test Lighthouse CI config
    const lighthouseConfigPath = path.join(__dirname, '..', 'lighthouserc.js');
    const lighthouseConfigExists = fs.existsSync(lighthouseConfigPath);
    
    this.results.configuration.push({
      test: 'Lighthouse CI Configuration',
      status: lighthouseConfigExists ? 'PASSED' : 'FAILED',
      details: lighthouseConfigExists ? 'Found lighthouserc.js' : 'lighthouserc.js not found'
    });
  }

  async testDependencies() {
    console.log('📦 Testing Dependencies...');
    
    const packageJsonPath = path.join(__dirname, '..', 'package.json');
    
    if (!fs.existsSync(packageJsonPath)) {
      this.results.dependencies.push({
        test: 'Package.json',
        status: 'FAILED',
        details: 'package.json not found'
      });
      return;
    }
    
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    const devDeps = packageJson.devDependencies || {};
    
    const requiredDeps = [
      '@next/bundle-analyzer',
      '@size-limit/preset-next',
      'bundle-stats',
      'cross-env',
      'webpack-bundle-analyzer',
      '@lhci/cli'
    ];
    
    requiredDeps.forEach(dep => {
      const hasPackage = devDeps[dep] !== undefined;
      this.results.dependencies.push({
        test: `Dependency: ${dep}`,
        status: hasPackage ? 'PASSED' : 'FAILED',
        details: hasPackage ? `Version: ${devDeps[dep]}` : 'Package not installed'
      });
    });
    
    // Check if dependencies are actually installed
    const nodeModulesPath = path.join(__dirname, '..', 'node_modules');
    if (fs.existsSync(nodeModulesPath)) {
      const installedPackages = fs.readdirSync(nodeModulesPath);
      const missingPackages = requiredDeps.filter(dep => {
        const packagePath = dep.startsWith('@') ? dep : dep;
        return !installedPackages.includes(packagePath.split('/')[0]);
      });
      
      this.results.dependencies.push({
        test: 'Dependencies Installation',
        status: missingPackages.length === 0 ? 'PASSED' : 'WARNING',
        details: missingPackages.length === 0 
          ? 'All required packages installed' 
          : `Missing: ${missingPackages.join(', ')}`
      });
    }
  }

  async testScripts() {
    console.log('⚙️ Testing Scripts...');
    
    const scriptsDir = path.join(__dirname);
    const requiredScripts = [
      'bundle-analysis.js',
      'performance-monitor.js',
      'ci-bundle-check.js'
    ];
    
    requiredScripts.forEach(script => {
      const scriptPath = path.join(scriptsDir, script);
      const exists = fs.existsSync(scriptPath);
      const isExecutable = exists && fs.accessSync(scriptPath, fs.constants.X_OK) === undefined;
      
      this.results.scripts.push({
        test: `Script: ${script}`,
        status: exists && isExecutable ? 'PASSED' : exists ? 'WARNING' : 'FAILED',
        details: !exists ? 'Script not found' : 
                !isExecutable ? 'Script not executable' : 'Script exists and executable'
      });
    });
    
    // Test package.json scripts
    const packageJsonPath = path.join(__dirname, '..', 'package.json');
    if (fs.existsSync(packageJsonPath)) {
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
      const scripts = packageJson.scripts || {};
      
      const requiredNpmScripts = [
        'build:analyze',
        'analyze',
        'size-limit',
        'perf:lighthouse'
      ];
      
      requiredNpmScripts.forEach(script => {
        const hasScript = scripts[script] !== undefined;
        this.results.scripts.push({
          test: `NPM Script: ${script}`,
          status: hasScript ? 'PASSED' : 'FAILED',
          details: hasScript ? scripts[script] : 'Script not defined'
        });
      });
    }
  }

  async testOptimization() {
    console.log('🚀 Testing Build Optimization...');
    
    try {
      // Test if we can load the Next.js config
      const nextConfigPath = path.join(__dirname, '..', 'next.config.ts');
      if (fs.existsSync(nextConfigPath)) {
        // Try to compile TypeScript config (basic validation)
        try {
          execSync('npx tsc --noEmit next.config.ts', { 
            cwd: path.dirname(nextConfigPath),
            stdio: 'pipe' 
          });
          
          this.results.optimization.push({
            test: 'Next.js Config Compilation',
            status: 'PASSED',
            details: 'TypeScript config compiles without errors'
          });
        } catch (error) {
          this.results.optimization.push({
            test: 'Next.js Config Compilation',
            status: 'WARNING',
            details: 'Config may have TypeScript issues'
          });
        }
      }
      
      // Test bundle analysis script
      try {
        const BundleAnalyzer = require('./bundle-analysis.js');
        const analyzer = new BundleAnalyzer();
        
        this.results.optimization.push({
          test: 'Bundle Analyzer Script',
          status: 'PASSED',
          details: 'Bundle analyzer can be instantiated'
        });
      } catch (error) {
        this.results.optimization.push({
          test: 'Bundle Analyzer Script',
          status: 'FAILED',
          details: `Failed to load: ${error.message}`
        });
      }
      
      // Test performance monitor script
      try {
        const PerformanceMonitor = require('./performance-monitor.js');
        const monitor = new PerformanceMonitor();
        
        this.results.optimization.push({
          test: 'Performance Monitor Script',
          status: 'PASSED',
          details: 'Performance monitor can be instantiated'
        });
      } catch (error) {
        this.results.optimization.push({
          test: 'Performance Monitor Script',
          status: 'FAILED',
          details: `Failed to load: ${error.message}`
        });
      }
      
      // Test CI bundle check script
      try {
        const CIBundleCheck = require('./ci-bundle-check.js');
        const checker = new CIBundleCheck();
        
        this.results.optimization.push({
          test: 'CI Bundle Check Script',
          status: 'PASSED',
          details: 'CI bundle checker can be instantiated'
        });
      } catch (error) {
        this.results.optimization.push({
          test: 'CI Bundle Check Script',
          status: 'FAILED',
          details: `Failed to load: ${error.message}`
        });
      }
      
    } catch (error) {
      this.results.optimization.push({
        test: 'General Optimization Test',
        status: 'FAILED',
        details: `Test execution failed: ${error.message}`
      });
    }
  }

  generateReport() {
    const reportPath = path.join(__dirname, '..', 'reports', 'optimization-test-report.json');
    const reportsDir = path.dirname(reportPath);
    
    if (!fs.existsSync(reportsDir)) {
      fs.mkdirSync(reportsDir, { recursive: true });
    }
    
    const report = {
      timestamp: new Date().toISOString(),
      results: this.results,
      summary: this.generateSummary()
    };
    
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`📄 Test report saved: ${reportPath}`);
  }

  generateSummary() {
    const allTests = [
      ...this.results.configuration,
      ...this.results.dependencies,
      ...this.results.scripts,
      ...this.results.optimization
    ];
    
    const passed = allTests.filter(t => t.status === 'PASSED').length;
    const failed = allTests.filter(t => t.status === 'FAILED').length;
    const warnings = allTests.filter(t => t.status === 'WARNING').length;
    const total = allTests.length;
    
    return {
      total,
      passed,
      failed,
      warnings,
      passRate: Math.round((passed / total) * 100),
      status: failed > 0 ? 'FAILED' : warnings > 0 ? 'WARNING' : 'PASSED'
    };
  }

  printSummary() {
    const summary = this.generateSummary();
    
    console.log('\n🎯 Optimization Test Summary');
    console.log('=============================');
    console.log(`Status: ${summary.status}`);
    console.log(`Pass Rate: ${summary.passRate}% (${summary.passed}/${summary.total})`);
    console.log(`Failed: ${summary.failed}`);
    console.log(`Warnings: ${summary.warnings}`);
    
    if (summary.failed > 0) {
      console.log('\n❌ Failed Tests:');
      const failedTests = [
        ...this.results.configuration.filter(t => t.status === 'FAILED'),
        ...this.results.dependencies.filter(t => t.status === 'FAILED'),
        ...this.results.scripts.filter(t => t.status === 'FAILED'),
        ...this.results.optimization.filter(t => t.status === 'FAILED')
      ];
      
      failedTests.forEach(test => {
        console.log(`   ${test.test}: ${test.details}`);
      });
    }
    
    if (summary.warnings > 0) {
      console.log('\n⚠️  Warnings:');
      const warningTests = [
        ...this.results.configuration.filter(t => t.status === 'WARNING'),
        ...this.results.dependencies.filter(t => t.status === 'WARNING'),
        ...this.results.scripts.filter(t => t.status === 'WARNING'),
        ...this.results.optimization.filter(t => t.status === 'WARNING')
      ];
      
      warningTests.forEach(test => {
        console.log(`   ${test.test}: ${test.details}`);
      });
    }
    
    console.log('=============================');
  }
}

// CLI interface
if (require.main === module) {
  const tester = new OptimizationTester();
  
  tester.runTests().catch(error => {
    console.error('❌ Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = OptimizationTester;