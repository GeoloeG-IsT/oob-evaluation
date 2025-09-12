# ML Evaluation Platform - Documentation Makefile
# 
# This Makefile provides convenient commands for generating and managing
# API documentation.

.PHONY: help docs clean serve install-deps check-deps

# Default target
help: ## Show this help message
	@echo "ML Evaluation Platform - Documentation Commands"
	@echo "============================================="
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

docs: check-deps ## Generate comprehensive API documentation
	@echo "🚀 Generating API documentation..."
	@python doc_generator.py
	@echo "✅ Documentation generated successfully!"

clean: ## Clean generated documentation files
	@echo "🧹 Cleaning documentation files..."
	@rm -rf api/*.html api/*.md api/*.json
	@rm -rf guides/*.md
	@rm -rf examples/*.py examples/*.sh examples/*.json
	@echo "✅ Documentation files cleaned!"

serve: docs ## Generate docs and serve locally on port 8080
	@echo "🌐 Starting local documentation server..."
	@echo "📖 Documentation available at: http://localhost:8080/api/"
	@cd .. && python -m http.server 8080 --directory docs

serve-dev: docs ## Generate docs and serve with auto-reload
	@echo "🌐 Starting development server with auto-reload..."
	@if command -v fswatch >/dev/null 2>&1; then \
		echo "👀 Watching for file changes..."; \
		fswatch -o . | xargs -n1 -I{} make docs & \
		cd .. && python -m http.server 8080 --directory docs; \
	else \
		echo "⚠️  fswatch not found. Install it for auto-reload functionality."; \
		make serve; \
	fi

install-deps: ## Install required Python dependencies
	@echo "📦 Installing Python dependencies..."
	@pip install pyyaml
	@echo "✅ Dependencies installed!"

check-deps: ## Check if required dependencies are available
	@python -c "import yaml, json, pathlib" 2>/dev/null || { \
		echo "❌ Missing dependencies. Run 'make install-deps' first."; \
		exit 1; \
	}

lint: ## Check documentation files for common issues
	@echo "🔍 Linting documentation files..."
	@# Check for broken internal links (simplified)
	@find . -name "*.md" -exec grep -l "](.*\.md)" {} \; | while read file; do \
		echo "Checking links in $$file..."; \
		grep -o "](.*\.md)" "$$file" | sed 's/](\(.*\))/\1/' | while read link; do \
			if [ ! -f "$$link" ]; then \
				echo "⚠️  Broken link in $$file: $$link"; \
			fi; \
		done; \
	done
	@echo "✅ Linting completed!"

stats: ## Show documentation statistics
	@echo "📊 Documentation Statistics"
	@echo "=========================="
	@echo "Generated files:"
	@find api guides examples -type f 2>/dev/null | wc -l | xargs -I {} echo "  Total files: {}"
	@find . -name "*.md" | wc -l | xargs -I {} echo "  Markdown files: {}"
	@find . -name "*.html" | wc -l | xargs -I {} echo "  HTML files: {}"
	@find . -name "*.json" | wc -l | xargs -I {} echo "  JSON files: {}"
	@echo
	@echo "Documentation size:"
	@du -sh . 2>/dev/null | echo "  Total size: $(cut -f1)"
	@echo
	@echo "Last generated:"
	@find api guides examples -name "*.md" -o -name "*.html" -o -name "*.json" 2>/dev/null | \
		xargs ls -lt | head -1 | awk '{print "  " $$6, $$7, $$8}' 2>/dev/null || echo "  No files found"

validate: check-deps ## Validate generated documentation
	@echo "🔍 Validating documentation..."
	@# Check OpenAPI JSON is valid
	@if [ -f "api/openapi.json" ]; then \
		python -c "import json; json.load(open('api/openapi.json'))" && \
		echo "✅ OpenAPI JSON is valid"; \
	else \
		echo "❌ OpenAPI JSON not found"; \
	fi
	@# Check HTML files are well-formed (basic check)
	@find api -name "*.html" | while read file; do \
		if grep -q "<html>" "$$file" && grep -q "</html>" "$$file"; then \
			echo "✅ $$file appears well-formed"; \
		else \
			echo "⚠️  $$file may have issues"; \
		fi; \
	done

watch: ## Watch for changes and regenerate docs automatically (requires fswatch)
	@if command -v fswatch >/dev/null 2>&1; then \
		echo "👀 Watching for changes to OpenAPI spec..."; \
		fswatch -o ../../specs/001-oob-evaluation-claude/contracts/api-spec.yaml | \
		xargs -n1 -I{} make docs; \
	else \
		echo "❌ fswatch not found. Install it with:"; \
		echo "   macOS: brew install fswatch"; \
		echo "   Linux: sudo apt-get install fswatch"; \
	fi

# Development helpers
dev-setup: install-deps docs ## Set up development environment
	@echo "🛠️  Development environment ready!"
	@echo "   Run 'make serve' to start local server"
	@echo "   Run 'make watch' to auto-regenerate on changes"

publish-check: validate ## Check if documentation is ready for publishing
	@echo "🔍 Checking documentation for publishing..."
	@# Check required files exist
	@required_files="api/index.html api/README.md api/openapi.json guides/quick-start.md"; \
	for file in $$required_files; do \
		if [ -f "$$file" ]; then \
			echo "✅ $$file exists"; \
		else \
			echo "❌ $$file missing"; \
			exit 1; \
		fi; \
	done
	@echo "✅ Documentation is ready for publishing!"

# Show current working directory (useful for debugging)
pwd:
	@echo "Current directory: $$(pwd)"
	@echo "Script directory: $$(dirname $$(readlink -f $(MAKEFILE_LIST)))"