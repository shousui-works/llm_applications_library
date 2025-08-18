# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Automatic version management system
- GitHub Actions workflow for automatic tag creation on main branch merge
- Dynamic version configuration using `__version__.py`

## [0.1.0] - 2024-08-18

### Added
- Initial release of LLM Applications Library
- OpenAI Vision API integration with retry functionality
- Haystack framework integration
- Utilities module with comprehensive functionality:
  - File I/O operations (YAML, text files)
  - Logging configuration utilities
  - Token counting and text splitting utilities (tiktoken)
  - PDF manipulation utilities (PyMuPDF, GCS support)
- Google Cloud Storage client implementation
- Comprehensive test suite with 90% coverage
- CI/CD pipeline with automated testing, linting, and formatting
- Package configuration for pip installation

### Features
- **OpenAI Integration**: Custom generator with vision capabilities and automatic retry
- **Haystack Components**: Standard AI pipeline building blocks
- **Utilities Module**: Complete set of helper functions for LLM applications
- **GCS Support**: Cloud storage integration for file operations
- **Type Safety**: Pydantic-based configuration management
- **Testing**: Comprehensive test coverage with pytest and tox
- **Code Quality**: Automated formatting and linting with ruff

### Technical Details
- Python 3.12+ requirement
- Modern packaging with pyproject.toml
- UV-based dependency management
- Automated CI/CD with GitHub Actions
- Semantic versioning support