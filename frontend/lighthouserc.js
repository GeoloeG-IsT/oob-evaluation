/**
 * Lighthouse CI Configuration for ML Evaluation Platform Frontend
 * 
 * This configuration sets up automated Lighthouse performance testing
 * for the Next.js application with specific settings for the ML platform.
 */

module.exports = {
  ci: {
    // Build configuration
    collect: {
      // Static site build for Lighthouse
      staticDistDir: './.next',
      
      // URLs to audit
      url: [
        'http://localhost:3000/',
        'http://localhost:3000/upload',
        'http://localhost:3000/annotations',
        'http://localhost:3000/models'
      ],
      
      // Number of runs per URL for more reliable results
      numberOfRuns: 3,
      
      // Lighthouse settings
      settings: {
        // Simulate slower CPU for more realistic testing
        throttling: {
          cpuSlowdownMultiplier: 2
        },
        
        // Audit specific categories
        onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
        
        // Skip certain audits that may not be relevant
        skipAudits: [
          'canonical', // May not be needed for all pages
          'crawlable-anchors' // Internal navigation may use different patterns
        ],
        
        // Additional configuration for ML platform
        emulatedFormFactor: 'desktop', // Primary usage expected on desktop
        throttlingMethod: 'devtools',
        
        // Longer timeout for potential ML operations
        maxWaitForFcp: 15 * 1000,
        maxWaitForLoad: 45 * 1000
      }
    },
    
    // Performance budgets
    assert: {
      assertions: {
        // Performance metrics
        'categories:performance': ['warn', { minScore: 0.8 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['warn', { minScore: 0.85 }],
        'categories:seo': ['warn', { minScore: 0.8 }],
        
        // Core Web Vitals
        'metrics:first-contentful-paint': ['warn', { maxNumericValue: 2000 }],
        'metrics:largest-contentful-paint': ['error', { maxNumericValue: 4000 }],
        'metrics:cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'metrics:total-blocking-time': ['warn', { maxNumericValue: 300 }],
        
        // Resource efficiency
        'metrics:speed-index': ['warn', { maxNumericValue: 3500 }],
        'metrics:interactive': ['warn', { maxNumericValue: 5000 }],
        
        // Bundle size related audits
        'audits:unused-css-rules': ['warn', { maxNumericValue: 20000 }],
        'audits:unused-javascript': ['warn', { maxNumericValue: 40000 }],
        'audits:unminified-css': 'error',
        'audits:unminified-javascript': 'error',
        'audits:efficient-animated-content': 'warn',
        
        // Image optimization (important for ML platform with image uploads)
        'audits:uses-optimized-images': 'warn',
        'audits:uses-webp-images': 'warn',
        'audits:uses-responsive-images': 'warn',
        'audits:properly-size-images': 'warn',
        
        // Network efficiency
        'audits:render-blocking-resources': ['warn', { maxNumericValue: 500 }],
        'audits:uses-text-compression': 'error',
        'audits:uses-rel-preconnect': 'warn',
        
        // JavaScript performance
        'audits:bootup-time': ['warn', { maxNumericValue: 3000 }],
        'audits:mainthread-work-breakdown': ['warn', { maxNumericValue: 4000 }],
        'audits:dom-size': ['warn', { maxNumericValue: 1500 }]
      }
    },
    
    // Upload results to Lighthouse CI server or temporary public storage
    upload: {
      target: 'temporary-public-storage',
      
      // Alternative: Use filesystem storage for CI artifacts
      // target: 'filesystem',
      // outputDir: './lighthouse-reports'
    },
    
    // Server configuration (if using LHCI server)
    server: {
      // port: 9001,
      // storage: {
      //   storageMethod: 'sql',
      //   sqlDialect: 'sqlite',
      //   sqlDatabasePath: './lighthouse-server.db'
      // }
    },
    
    // Wizard configuration for setup
    wizard: {
      // Skip wizard in CI environments
      skipWizard: process.env.CI === 'true'
    }
  },
  
  // Custom performance budgets for different page types
  budgets: [
    {
      // Homepage budget
      path: '/',
      resourceCounts: [
        { resourceType: 'script', budget: 15 },
        { resourceType: 'stylesheet', budget: 5 },
        { resourceType: 'image', budget: 10 },
        { resourceType: 'font', budget: 3 }
      ],
      resourceSizes: [
        { resourceType: 'script', budget: 600 }, // 600KB JS
        { resourceType: 'stylesheet', budget: 50 }, // 50KB CSS
        { resourceType: 'image', budget: 200 }, // 200KB images
        { resourceType: 'font', budget: 100 }, // 100KB fonts
        { resourceType: 'total', budget: 1000 } // 1MB total
      ]
    },
    
    {
      // Upload page budget (may have additional ML-related scripts)
      path: '/upload',
      resourceCounts: [
        { resourceType: 'script', budget: 20 },
        { resourceType: 'stylesheet', budget: 6 }
      ],
      resourceSizes: [
        { resourceType: 'script', budget: 800 }, // 800KB JS (ML libraries)
        { resourceType: 'stylesheet', budget: 60 }, // 60KB CSS
        { resourceType: 'total', budget: 1200 } // 1.2MB total
      ]
    }
  ]
};