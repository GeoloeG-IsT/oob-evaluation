    async def run_comprehensive_validation(self) -> str:
        """Execute the complete T085 validation workflow"""
        logger.info("=" * 80)
        logger.info("Starting T085 ML Evaluation Platform Comprehensive Validation")
        logger.info("=" * 80)
        
        try:
            # Setup validation environment
            await self.setup()
            
            # Start Docker Compose services
            logger.info("Phase 1: Starting Docker Compose services...")
            service_status = await self.docker_manager.start_services()
            self.report.service_status = service_status
            
            # Log service status
            for name, status in service_status.items():
                if status.ready:
                    logger.info(f"✓ {name}: Ready")
                else:
                    logger.warning(f"✗ {name}: Not ready - {status.status}")
            
            # Phase 2: Execute all validation workflows
            logger.info("Phase 2: Executing validation workflows...")
            
            validation_workflows = [
                ("Service Health Validation", self.validate_service_health),
                ("Step 1: Upload and Organize Images", self.validate_step_1_upload_images),
                ("Step 2: Manual Annotation", self.validate_step_2_manual_annotation),
                ("Step 3: Model Selection and Assisted Annotation", self.validate_step_3_assisted_annotation),
                ("Step 4: Model Inference (Single and Batch)", self.validate_step_4_model_inference),
                ("Step 5: Performance Evaluation", self.validate_step_5_performance_evaluation),
                ("Step 6: Model Training/Fine-tuning", self.validate_step_6_model_training),
                ("Step 7: Model Deployment", self.validate_step_7_model_deployment),
                ("Step 8: Data Export", self.validate_step_8_data_export),
                ("Error Handling Validation", self.validate_error_handling),
                ("Performance Requirements Validation", self.validate_performance_requirements)
            ]
            
            for workflow_name, workflow_func in validation_workflows:
                logger.info(f"Executing: {workflow_name}")
                start_time = time.time()
                
                try:
                    result = await workflow_func()
                    result.workflow = workflow_name
                    result.duration = time.time() - start_time
                    
                    self.report.results.append(result)
                    
                    if result.success:
                        logger.info(f"✓ {workflow_name} - PASSED ({result.duration:.2f}s)")
                    else:
                        logger.warning(f"✗ {workflow_name} - FAILED ({result.duration:.2f}s): {result.message}")
                        for error in result.errors:
                            logger.error(f"    Error: {error}")
                
                except Exception as e:
                    error_result = ValidationResult(
                        step=workflow_name,
                        workflow=workflow_name,
                        success=False,
                        message=f"Workflow execution failed: {str(e)}",
                        duration=time.time() - start_time,
                        errors=[str(e)]
                    )
                    self.report.results.append(error_result)
                    logger.error(f"✗ {workflow_name} - EXCEPTION: {e}")
            
            # Phase 3: Collect service logs for debugging
            logger.info("Phase 3: Collecting service logs...")
            for service_name in self.docker_manager.services:
                logs = await self.docker_manager.get_service_logs(service_name, tail_lines=50)
                self.report.docker_compose_logs[service_name] = logs
            
            # Phase 4: Generate comprehensive report
            logger.info("Phase 4: Generating comprehensive report...")
            return await self.generate_comprehensive_report()
            
        finally:
            # Cleanup
            await self.cleanup()
            # Note: Not stopping services automatically to allow manual inspection
            # Use docker-compose down manually or call stop_services() if needed

class WorkflowValidationMixin:
    """Mixin class containing all workflow validation methods"""
    
    async def validate_service_health(self) -> ValidationResult:
        """Validate comprehensive service health including API endpoints"""
        errors = []
        service_checks = {}
        api_calls = []
        
        logger.info("Validating comprehensive service health...")
        
        # Check all service containers
        for service_name, service_status in self.report.service_status.items():
            service_checks[service_name] = {
                'container_ready': service_status.ready,
                'status': service_status.status,
                'health': service_status.health
            }
            
            if not service_status.ready:
                errors.append(f"{service_name} service not ready: {service_status.status}")
        
        # API endpoint health checks
        api_endpoints = [
            ("Backend Health", f"{BACKEND_URL}/health"),
            ("Backend API Docs", f"{BACKEND_URL}/docs"),
            ("Backend OpenAPI", f"{BACKEND_URL}/openapi.json"),
            ("Frontend", FRONTEND_URL),
            ("Flower Monitor", FLOWER_URL)
        ]
        
        for endpoint_name, url in api_endpoints:
            try:
                start_time = time.time()
                async with self.session.get(url) as response:
                    response_time = time.time() - start_time
                    api_call = {
                        'endpoint': endpoint_name,
                        'url': url,
                        'status_code': response.status,
                        'response_time': response_time,
                        'success': response.status == 200
                    }
                    api_calls.append(api_call)
                    
                    if response.status != 200:
                        errors.append(f"{endpoint_name} returned status {response.status}")
                    else:
                        service_checks[endpoint_name.lower().replace(' ', '_')] = {
                            'status_code': response.status,
                            'response_time': response_time
                        }
            
            except Exception as e:
                errors.append(f"{endpoint_name} connection failed: {str(e)}")
                api_calls.append({
                    'endpoint': endpoint_name,
                    'url': url,
                    'error': str(e),
                    'success': False
                })
        
        # Database connectivity test
        try:
            if not self.db_pool:
                self.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            
            async with self.db_pool.acquire() as conn:
                # Test basic query
                result = await conn.fetchval("SELECT version()")
                service_checks['database_query'] = {'version': result}
                
                # Test if required tables exist (basic schema check)
                tables = await conn.fetch("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                service_checks['database_schema'] = {'table_count': len(tables)}
                
        except Exception as e:
            errors.append(f"Database connectivity test failed: {str(e)}")
        
        # Redis connectivity test
        try:
            if not self.redis_client:
                self.redis_client = redis.from_url(REDIS_URL)
            
            info = self.redis_client.info()
            service_checks['redis'] = {
                'version': info.get('redis_version'),
                'connected_clients': info.get('connected_clients')
            }
            
        except Exception as e:
            errors.append(f"Redis connectivity test failed: {str(e)}")
        
        success = len(errors) == 0
        message = "All services healthy and responsive" if success else f"Found {len(errors)} service issues"
        
        return ValidationResult(
            step="Service Health Validation",
            workflow="Infrastructure",
            success=success,
            message=message,
            duration=0,  # Will be set by caller
            data=service_checks,
            errors=errors,
            api_calls=api_calls
        )

    # Import validation methods from separate modules
    async def validate_step_1_upload_images(self) -> ValidationResult:
        from t085_workflow_validations import WorkflowValidations
        validator = WorkflowValidations(self.session, BACKEND_URL, self.test_data)
        return await validator.validate_step_1_upload_images()
    
    async def validate_step_2_manual_annotation(self) -> ValidationResult:
        from t085_workflow_validations import WorkflowValidations
        validator = WorkflowValidations(self.session, BACKEND_URL, self.test_data)
        return await validator.validate_step_2_manual_annotation()
    
    async def validate_step_3_assisted_annotation(self) -> ValidationResult:
        from t085_workflow_validations import WorkflowValidations
        validator = WorkflowValidations(self.session, BACKEND_URL, self.test_data)
        return await validator.validate_step_3_assisted_annotation()
    
    async def validate_step_4_model_inference(self) -> ValidationResult:
        from t085_workflow_validations import WorkflowValidations
        validator = WorkflowValidations(self.session, BACKEND_URL, self.test_data)
        return await validator.validate_step_4_model_inference()
    
    async def validate_step_5_performance_evaluation(self) -> ValidationResult:
        from t085_workflow_validations import WorkflowValidations
        validator = WorkflowValidations(self.session, BACKEND_URL, self.test_data)
        return await validator.validate_step_5_performance_evaluation()
    
    async def validate_step_6_model_training(self) -> ValidationResult:
        from t085_workflow_validations import WorkflowValidations
        validator = WorkflowValidations(self.session, BACKEND_URL, self.test_data)
        return await validator.validate_step_6_model_training()
    
    async def validate_step_7_model_deployment(self) -> ValidationResult:
        from t085_workflow_validations import WorkflowValidations
        validator = WorkflowValidations(self.session, BACKEND_URL, self.test_data)
        return await validator.validate_step_7_model_deployment()
    
    async def validate_step_8_data_export(self) -> ValidationResult:
        from t085_workflow_validations import WorkflowValidations
        validator = WorkflowValidations(self.session, BACKEND_URL, self.test_data)
        return await validator.validate_step_8_data_export()
    
    async def validate_error_handling(self) -> ValidationResult:
        from t085_error_validation import ErrorHandlingValidation
        validator = ErrorHandlingValidation(self.session, BACKEND_URL, BASE_DIR)
        return await validator.validate_error_handling()
    
    async def validate_performance_requirements(self) -> ValidationResult:
        from t085_error_validation import PerformanceValidation
        validator = PerformanceValidation(self.session, BACKEND_URL)
        return await validator.validate_performance_requirements()
    
    async def generate_comprehensive_report(self) -> str:
        """Generate comprehensive validation report with all details"""
        self.report.end_time = datetime.now()
        total_duration = (self.report.end_time - self.report.start_time).total_seconds()
        
        # Calculate summary statistics
        successful_steps = sum(1 for r in self.report.results if r.success)
        total_steps = len(self.report.results)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Group results by workflow
        workflow_results = {}
        for result in self.report.results:
            workflow = result.workflow
            if workflow not in workflow_results:
                workflow_results[workflow] = []
            workflow_results[workflow].append(result)
        
        # Calculate performance metrics summary
        all_api_calls = []
        all_performance_metrics = {}
        total_errors = 0
        total_warnings = 0
        
        for result in self.report.results:
            all_api_calls.extend(result.api_calls)
            all_performance_metrics.update(result.performance_metrics)
            total_errors += len(result.errors)
            total_warnings += len(result.warnings)
        
        # Calculate API performance statistics
        if all_api_calls:
            response_times = [call.get('response_time', 0) for call in all_api_calls if 'response_time' in call]
            self.report.performance_summary = {
                'total_api_calls': len(all_api_calls),
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'successful_api_calls': sum(1 for call in all_api_calls if call.get('success', False)),
                'failed_api_calls': sum(1 for call in all_api_calls if not call.get('success', True))
            }
        
        # Generate recommendations
        self.generate_recommendations()
        
        self.report.summary = {
            'total_duration': total_duration,
            'successful_steps': successful_steps,
            'total_steps': total_steps,
            'success_rate': success_rate,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'workflows_tested': len(workflow_results),
            'services_tested': len(self.report.service_status),
            'docker_compose_used': True
        }
        
        # Generate report text
        report_lines = [
            "=" * 100,
            "T085 - ML EVALUATION PLATFORM COMPREHENSIVE VALIDATION REPORT",
            "=" * 100,
            f"Generated: {self.report.end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)",
            f"Success Rate: {success_rate:.1f}% ({successful_steps}/{total_steps} workflows passed)",
            f"Total Issues: {total_errors} errors, {total_warnings} warnings",
            "",
            "EXECUTIVE SUMMARY:",
            "-" * 50
        ]
        
        # Add executive summary
        if success_rate >= 90:
            report_lines.append("✅ EXCELLENT: Platform validation highly successful")
        elif success_rate >= 75:
            report_lines.append("✅ GOOD: Platform validation mostly successful with minor issues")
        elif success_rate >= 50:
            report_lines.append("⚠️  MODERATE: Platform validation partially successful with significant issues")
        else:
            report_lines.append("❌ CRITICAL: Platform validation failed with major issues")
        
        report_lines.extend([
            "",
            "DOCKER COMPOSE SERVICES:",
            "-" * 50
        ])
        
        # Service status summary
        for service_name, status in self.report.service_status.items():
            status_icon = "✅" if status.ready else "❌"
            report_lines.append(f"{status_icon} {service_name.ljust(20)}: {status.status} ({status.health})")
            if status.error_message:
                report_lines.append(f"    Error: {status.error_message}")
        
        # API Performance Summary
        if self.report.performance_summary:
            perf = self.report.performance_summary
            report_lines.extend([
                "",
                "API PERFORMANCE SUMMARY:",
                "-" * 50,
                f"Total API Calls: {perf['total_api_calls']} ({perf['successful_api_calls']} successful, {perf['failed_api_calls']} failed)",
                f"Average Response Time: {perf['avg_response_time']:.3f}s",
                f"Response Time Range: {perf['min_response_time']:.3f}s - {perf['max_response_time']:.3f}s"
            ])
        
        # Workflow Results
        report_lines.extend([
            "",
            "DETAILED WORKFLOW RESULTS:",
            "-" * 50
        ])
        
        for workflow_name, results in workflow_results.items():
            workflow_success = all(r.success for r in results)
            workflow_icon = "✅" if workflow_success else "❌"
            report_lines.append(f"\n{workflow_icon} {workflow_name.upper()} WORKFLOW:")
            
            for result in results:
                step_icon = "  ✓" if result.success else "  ✗"
                report_lines.append(f"{step_icon} {result.step}")
                report_lines.append(f"      Duration: {result.duration:.2f}s")
                report_lines.append(f"      Message: {result.message}")
                
                if result.errors:
                    report_lines.append(f"      Errors ({len(result.errors)}):")
                    for error in result.errors[:3]:  # Limit to first 3 errors
                        report_lines.append(f"        - {error}")
                    if len(result.errors) > 3:
                        report_lines.append(f"        ... and {len(result.errors) - 3} more errors")
                
                if result.warnings:
                    report_lines.append(f"      Warnings ({len(result.warnings)}):")
                    for warning in result.warnings[:2]:  # Limit to first 2 warnings
                        report_lines.append(f"        - {warning}")
                
                # Add key performance metrics for this step
                if result.performance_metrics:
                    report_lines.append(f"      Performance:")
                    for metric_name, metric_value in result.performance_metrics.items():
                        if isinstance(metric_value, float):
                            report_lines.append(f"        {metric_name}: {metric_value:.3f}")
                        else:
                            report_lines.append(f"        {metric_name}: {metric_value}")
        
        # Recommendations
        if self.report.recommendations:
            report_lines.extend([
                "",
                "RECOMMENDATIONS:",
                "-" * 50
            ])
            for i, rec in enumerate(self.report.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        # Next Steps
        report_lines.extend([
            "",
            "NEXT STEPS:",
            "-" * 50
        ])
        
        if success_rate >= 90:
            report_lines.extend([
                "✅ Platform is ready for production deployment",
                "✅ All core workflows validated successfully",
                "📋 Consider performance optimization for scale",
                "📋 Add monitoring and alerting for production"
            ])
        elif success_rate >= 75:
            report_lines.extend([
                "⚠️  Address remaining issues before production deployment",
                "✅ Core functionality working well",
                "📋 Implement missing ML model features",
                "📋 Fix identified error handling gaps"
            ])
        else:
            report_lines.extend([
                "❌ Critical issues must be resolved before production",
                "🔧 Focus on service stability and API functionality",
                "🔧 Complete ML model integration",
                "🔧 Implement comprehensive error handling"
            ])
        
        report_lines.extend([
            "",
            "TECHNICAL DETAILS:",
            "-" * 50,
            f"Validation Environment: Docker Compose",
            f"Backend URL: {BACKEND_URL}",
            f"Frontend URL: {FRONTEND_URL}",
            f"Test Images Generated: {len(self.test_data.get('images', []))}",
            f"Total API Endpoints Tested: {len(set(call.get('endpoint', '') for call in all_api_calls))}",
            "",
            "=" * 100
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = BASE_DIR / f"t085_validation_report_{timestamp}.txt"
        report_file.write_text(report_text)
        
        # Save JSON report for programmatic access
        json_report = {
            'timestamp': self.report.end_time.isoformat(),
            'summary': self.report.summary,
            'service_status': {name: {
                'ready': status.ready,
                'status': status.status,
                'health': status.health,
                'error': status.error_message
            } for name, status in self.report.service_status.items()},
            'workflow_results': [{
                'step': r.step,
                'workflow': r.workflow,
                'success': r.success,
                'duration': r.duration,
                'message': r.message,
                'error_count': len(r.errors),
                'warning_count': len(r.warnings),
                'api_calls': len(r.api_calls),
                'performance_metrics': r.performance_metrics
            } for r in self.report.results],
            'performance_summary': self.report.performance_summary,
            'recommendations': self.report.recommendations
        }
        
        json_file = BASE_DIR / f"t085_validation_report_{timestamp}.json"
        json_file.write_text(json.dumps(json_report, indent=2))
        
        logger.info(f"Comprehensive validation report saved:")
        logger.info(f"  Text: {report_file}")
        logger.info(f"  JSON: {json_file}")
        
        return report_text
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Analyze service health
        unhealthy_services = [name for name, status in self.report.service_status.items() if not status.ready]
        if unhealthy_services:
            recommendations.append(f"Fix unhealthy services: {', '.join(unhealthy_services)}")
        
        # Analyze workflow failures
        failed_workflows = [r.workflow for r in self.report.results if not r.success]
        workflow_failures = {}
        for workflow in failed_workflows:
            workflow_failures[workflow] = workflow_failures.get(workflow, 0) + 1
        
        for workflow, count in workflow_failures.items():
            recommendations.append(f"Address {count} failed steps in {workflow} workflow")
        
        # Performance recommendations
        slow_api_calls = [call for call in sum([r.api_calls for r in self.report.results], []) 
                         if call.get('response_time', 0) > 2.0]
        if slow_api_calls:
            recommendations.append(f"Optimize {len(slow_api_calls)} slow API endpoints (>2s response time)")
        
        # Missing ML features
        ml_pending = [r for r in self.report.results if "ML" in r.message or "pending" in r.message.lower()]
        if ml_pending:
            recommendations.append("Implement missing ML model features for complete workflow validation")
        
        # Error handling improvements
        error_handling_issues = [r for r in self.report.results if r.workflow == "Error Handling" and not r.success]
        if error_handling_issues:
            recommendations.append("Improve error handling based on failed error scenarios")
        
        # Docker and infrastructure
        if any("docker" in str(r.errors).lower() for r in self.report.results):
            recommendations.append("Review Docker Compose configuration and container health checks")
        
        # Add standard production recommendations
        if not recommendations:  # If no specific issues found
            recommendations.extend([
                "Implement comprehensive monitoring and logging",
                "Add automated backup procedures for data persistence",
                "Configure CI/CD pipeline with this validation system",
                "Set up production environment with load balancing"
            ])
        
        self.report.recommendations = recommendations[:10]  # Limit to top 10 recommendations

# Main execution functions
async def main():
    """Main entry point for T085 comprehensive validation"""
    print("=" * 80)
    print("T085 - ML Evaluation Platform Comprehensive Validation System")
    print("=" * 80)
    print("This system will:")
    print("1. Start all Docker Compose services")
    print("2. Wait for all services to become healthy")
    print("3. Execute all 8 quickstart workflow validations")
    print("4. Test error handling scenarios")
    print("5. Validate performance requirements")
    print("6. Generate comprehensive reports")
    print("7. Provide actionable recommendations")
    print("\nStarting validation...\n")
    
    validator = T085ComprehensiveValidator()
    
    try:
        report = await validator.run_comprehensive_validation()
        
        print("\n" + "=" * 80)
        print("T085 COMPREHENSIVE VALIDATION COMPLETED")
        print("=" * 80)
        print(report)
        
        # Determine exit code based on results
        successful_steps = sum(1 for r in validator.report.results if r.success)
        total_steps = len(validator.report.results)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
        
        if success_rate >= 90:
            print("\n🎉 VALIDATION HIGHLY SUCCESSFUL! Platform ready for production.")
            sys.exit(0)
        elif success_rate >= 75:
            print("\n✅ VALIDATION MOSTLY SUCCESSFUL with minor issues to address.")
            sys.exit(0)
        elif success_rate >= 50:
            print("\n⚠️  VALIDATION PARTIALLY SUCCESSFUL with significant issues to resolve.")
            sys.exit(1)
        else:
            print("\n❌ VALIDATION FAILED with critical issues requiring immediate attention.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        print("\n⚠️  Validation interrupted. Cleaning up...")
        await validator.cleanup()
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("Check logs for details.")
        sys.exit(1)

def run_manual_validation():
    """Run validation with manual Docker Compose management"""
    print("=" * 80)
    print("T085 - MANUAL VALIDATION MODE")
    print("=" * 80)
    print("This mode assumes Docker Compose services are already running.")
    print("Please ensure services are started with: docker-compose up -d")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    
    asyncio.run(main())

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        run_manual_validation()
    else:
        asyncio.run(main())
