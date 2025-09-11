# Frontend Build Optimization Scripts

This directory contains automated scripts for bundle analysis, performance monitoring, and CI/CD integration for the ML Evaluation Platform frontend.

## Scripts Overview

### Core Analysis Scripts

**`bundle-analysis.js`** - Comprehensive bundle size analysis
- Analyzes JavaScript, CSS, and asset bundles
- Generates detailed HTML and JSON reports
- Checks size budgets and provides recommendations
- Tracks historical performance data

**`performance-monitor.js`** - Continuous performance monitoring
- Monitors build times and bundle sizes
- Detects performance regressions
- Generates trend analysis and recommendations
- Integrates with CI/CD for automated checks

**`ci-bundle-check.js`** - CI/CD-focused bundle validation
- Designed for automated pipeline integration
- Compares against baseline metrics
- Generates PR comments with bundle analysis
- Configurable fail conditions for builds

**`test-optimization.js`** - Validation and testing script
- Tests that all optimization configurations are working
- Validates dependencies and script setup
- Generates diagnostic reports
- Ensures build system integrity

## Usage

### Development Workflow

```bash
# Run comprehensive bundle analysis
npm run bundle:analyze

# Monitor performance during development
npm run bundle:monitor

# Test optimization setup
npm run test:optimization

# Quick size check
npm run size-limit
```

### CI/CD Integration

```bash
# Full CI bundle check with PR comments
npm run ci:bundle-check -- --pr-comment

# CI check with GitHub Actions format
npm run ci:bundle-check -- --format=github

# Run without failing build on budget violations
npm run ci:bundle-check -- --no-fail-budget
```

### Analysis and Reporting

```bash
# Generate bundle visualization
npm run analyze

# Run Lighthouse performance testing
npm run perf:lighthouse

# Generate detailed size reports
npm run size-limit:report
```

## Script Dependencies

All scripts require the following dependencies (installed via npm):
- `@next/bundle-analyzer` - Bundle analysis integration
- `webpack-bundle-analyzer` - Webpack bundle visualization
- `@size-limit/preset-next` - Bundle size limits
- `@lhci/cli` - Lighthouse CI integration
- `bundle-stats` - Bundle statistics generation
- `cross-env` - Cross-platform environment variables

## Configuration Files

Scripts work with the following configuration files:
- `next.config.ts` - Next.js build optimization
- `.size-limit.js` - Bundle size budgets
- `lighthouserc.js` - Lighthouse CI configuration
- `.github/workflows/bundle-analysis.yml` - GitHub Actions workflow

## Output Locations

Scripts generate reports in:
- `reports/bundle-stats/` - Bundle analysis reports
- `performance-data/` - Performance monitoring data
- `ci-reports/` - CI/CD specific reports
- `.lighthouseci/` - Lighthouse CI reports

## Common Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `npm run test:optimization` | Validate setup | Before first use, after config changes |
| `npm run bundle:analyze` | Full analysis | Weekly, before major releases |
| `npm run bundle:monitor` | Performance check | Daily, during development |
| `npm run ci:bundle-check` | CI validation | Automated in CI/CD pipelines |
| `npm run size-limit` | Quick size check | Before committing changes |

## Troubleshooting

### Common Issues

1. **Scripts not executable**
   ```bash
   chmod +x scripts/*.js
   ```

2. **Dependencies missing**
   ```bash
   npm install
   npm run test:optimization
   ```

3. **Build output not found**
   ```bash
   npm run build
   npm run bundle:analyze
   ```

4. **Permission errors in CI**
   - Ensure GitHub token has proper permissions
   - Check workflow permissions in repository settings

### Debug Mode

Most scripts support verbose output for debugging:

```bash
DEBUG=true npm run bundle:analyze
VERBOSE=true npm run bundle:monitor
```

## Integration Examples

### GitHub Actions Integration

```yaml
- name: Run bundle analysis
  run: npm run ci:bundle-check -- --format=github --pr-comment
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Local Development Integration

```bash
# Pre-commit hook
npm run size-limit || exit 1

# Pre-push hook
npm run bundle:monitor || exit 1
```

### Baseline Management

```bash
# Update baseline (run on main branch)
npm run ci:bundle-check -- --update-baseline

# Compare with specific baseline
npm run ci:bundle-check -- --baseline=develop
```

## Performance Budgets

Default budgets are defined in `.size-limit.js`:
- Main Bundle: 300KB (gzipped)
- Framework Bundle: 400KB (gzipped)
- Total JavaScript: 800KB (gzipped)
- CSS Bundle: 50KB (gzipped)

Budgets can be customized based on application requirements.

## Continuous Improvement

The optimization system includes:
- **Automated regression detection** - Prevents performance degradation
- **Historical tracking** - Long-term performance trends
- **Recommendation engine** - Suggests specific optimizations
- **CI/CD integration** - Blocks problematic changes before merge

For detailed information about the optimization system, see `docs/BUNDLE_OPTIMIZATION.md`.