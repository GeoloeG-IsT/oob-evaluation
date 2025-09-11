# ML Evaluation Platform - Frontend

Web interface for evaluating Object Detection and Segmentation models (YOLO11/12, RT-DETR, SAM2) with capabilities for image upload, annotation, model inference, training/fine-tuning, and performance evaluation.

## Tech Stack

- **Framework**: Next.js 15.5.3 with App Router
- **React**: 19.1.0
- **TypeScript**: 5.x
- **Styling**: Tailwind CSS v4
- **Build**: Turbopack
- **Performance Monitoring**: Lighthouse, Bundle Analyzer, Size Limit

## Features

### Core Functionality
- **Image Management**: Upload and organize images in train/val/test splits
- **Annotation Tools**: Draw bounding boxes and segments with assisted annotation (SAM2)
- **Model Inference**: Real-time and batch processing with progress monitoring
- **Training Interface**: Configure and monitor model fine-tuning jobs
- **Performance Metrics**: Visualize mAP, IoU, precision, recall, F1 scores
- **Model Deployment**: Manage deployed model endpoints
- **Data Export**: Export datasets in COCO, YOLO, Pascal VOC formats

### UI Components (To Be Implemented)
- Image upload with drag-and-drop
- Canvas-based annotation editor
- Real-time WebSocket updates for long-running tasks
- Interactive charts for metrics visualization
- Dataset browser with filtering and search
- Model configuration forms
- Training progress monitoring
- Deployment management dashboard

## Getting Started

### Prerequisites
- Node.js 18+
- npm 10.8.2+

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Development with Docker

The frontend is part of the Docker Compose setup:

```bash
# From project root
docker-compose -f docker-compose.development.yml up frontend
```

The frontend will be available at http://localhost:3000

### Environment Variables

Create a `.env.local` file:

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# App Configuration
NEXT_PUBLIC_APP_NAME="ML Evaluation Platform"
NEXT_PUBLIC_APP_VERSION=1.0.0

# Feature Flags (optional)
NEXT_PUBLIC_ENABLE_ASSISTED_ANNOTATION=true
NEXT_PUBLIC_ENABLE_MODEL_DEPLOYMENT=true
```

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   ├── images/            # Image management pages
│   ├── annotations/       # Annotation interface
│   ├── models/            # Model management
│   ├── training/          # Training interface
│   ├── evaluation/        # Performance metrics
│   └── deployments/       # Deployment management
├── src/
│   ├── components/        # React components
│   │   ├── ui/           # Base UI components
│   │   ├── canvas/       # Annotation canvas
│   │   ├── upload/       # Upload components
│   │   └── charts/       # Data visualization
│   ├── lib/              # Utility libraries
│   │   ├── api/          # API client
│   │   ├── websocket/    # WebSocket client
│   │   └── utils/        # Helper functions
│   ├── hooks/            # Custom React hooks
│   ├── types/            # TypeScript type definitions
│   └── styles/           # Global styles
├── public/               # Static assets
└── tests/               # Test files

```

## Available Scripts

```bash
# Development
npm run dev              # Start dev server with Turbopack
npm run lint            # Run ESLint
npm run type-check      # Run TypeScript compiler

# Build & Production
npm run build           # Build for production
npm run build:analyze   # Build with bundle analysis
npm run start           # Start production server

# Performance & Optimization
npm run size-limit      # Check bundle sizes
npm run perf:lighthouse # Run Lighthouse audit
npm run bundle:analyze  # Analyze bundle composition
npm run bundle:monitor  # Monitor performance metrics

# Testing
npm run test            # Run tests (to be implemented)
npm run test:e2e        # Run E2E tests (to be implemented)
```

## Performance Optimization

The frontend includes several performance optimization tools:

- **Bundle Analysis**: Webpack Bundle Analyzer for understanding bundle composition
- **Size Limits**: Automated bundle size checking with Size Limit
- **Lighthouse CI**: Performance auditing with Lighthouse
- **Code Splitting**: Automatic code splitting with Next.js
- **Image Optimization**: Next.js Image component for optimized loading

## API Integration

The frontend communicates with the backend API at `http://localhost:8000`. Key endpoints:

- `/api/v1/images` - Image management
- `/api/v1/annotations` - Annotation CRUD
- `/api/v1/models` - Model management
- `/api/v1/inference` - Inference jobs
- `/api/v1/training` - Training jobs
- `/api/v1/evaluation` - Performance metrics
- `/api/v1/deployments` - Model deployments

WebSocket connections for real-time updates:
- `/ws/training` - Training progress
- `/ws/inference` - Inference progress
- `/ws/notifications` - System notifications

## Contributing

1. Follow the existing code structure and conventions
2. Use TypeScript for all new code
3. Add proper type definitions
4. Follow the component structure in `src/components`
5. Use Tailwind CSS for styling
6. Ensure bundle size limits are met
7. Run lint and type checks before committing

## Troubleshooting

### Common Issues

1. **Port 3000 already in use**: Kill the process using the port or change the port in package.json
2. **API connection failed**: Ensure backend is running on port 8000
3. **WebSocket connection failed**: Check CORS settings and WebSocket URL
4. **Build failures**: Clear `.next` folder and node_modules, then reinstall

### Development Tips

- Use the React Developer Tools extension for debugging
- Monitor Network tab for API calls
- Check WebSocket connections in Network > WS tab
- Use Lighthouse for performance insights
- Regular bundle size checks with `npm run size-limit`

## License

Proprietary - All rights reserved
