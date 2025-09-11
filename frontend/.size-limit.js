/**
 * Size Limit Configuration for ML Evaluation Platform Frontend
 * 
 * This configuration defines bundle size budgets and limits
 * to ensure optimal performance for the ML evaluation application.
 */

module.exports = [
  // Main Application Bundle
  {
    name: "Main Bundle (JS)",
    path: ".next/static/chunks/main-*.js",
    limit: "300 KB",
    webpack: true,
    gzip: true
  },
  
  // Framework Bundle (React, Next.js core)
  {
    name: "Framework Bundle",
    path: ".next/static/chunks/framework-*.js", 
    limit: "400 KB",
    webpack: true,
    gzip: true
  },
  
  // Runtime Bundle (Webpack runtime)
  {
    name: "Webpack Runtime",
    path: ".next/static/chunks/webpack-*.js",
    limit: "50 KB",
    webpack: true,
    gzip: true
  },
  
  // Commons/Shared Bundle
  {
    name: "Commons Bundle",
    path: ".next/static/chunks/commons-*.js",
    limit: "200 KB",
    webpack: true,
    gzip: true
  },
  
  // Pages Bundle (Home page)
  {
    name: "Home Page",
    path: ".next/static/chunks/pages/index-*.js",
    limit: "150 KB",
    webpack: true,
    gzip: true
  },
  
  // App Bundle (_app.js)
  {
    name: "App Bundle",
    path: ".next/static/chunks/pages/_app-*.js", 
    limit: "250 KB",
    webpack: true,
    gzip: true
  },
  
  // CSS Bundles
  {
    name: "CSS Bundle (Total)",
    path: ".next/static/css/*.css",
    limit: "50 KB",
    webpack: false,
    gzip: true
  },
  
  // Vendor Libraries (when they get chunked separately)
  {
    name: "React Bundle",
    path: ".next/static/chunks/react-*.js",
    limit: "200 KB",
    webpack: true,
    gzip: true,
    ignore: ["**/node_modules/**"]
  },
  
  // Canvas/ML Libraries (future)
  {
    name: "Canvas Libraries",
    path: ".next/static/chunks/canvas-*.js",
    limit: "500 KB",
    webpack: true,
    gzip: true,
    ignore: ["**/node_modules/**"]
  },
  
  // Total Application Size
  {
    name: "Total JavaScript",
    path: ".next/static/chunks/*.js",
    limit: "800 KB",
    webpack: true,
    gzip: true
  },
  
  // First Load JS (Critical for performance)
  {
    name: "First Load JS",
    path: [
      ".next/static/chunks/framework-*.js",
      ".next/static/chunks/main-*.js", 
      ".next/static/chunks/webpack-*.js",
      ".next/static/chunks/pages/_app-*.js"
    ],
    limit: "600 KB",
    webpack: true,
    gzip: true
  }
];