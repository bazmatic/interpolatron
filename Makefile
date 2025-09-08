# Video Interpolation Tool Makefile

.PHONY: help install test clean

# Default target
help:
	@echo "Video Interpolation Tool - Available Commands"
	@echo "============================================="
	@echo ""
	@echo "Installation:"
	@echo "  install    - Make scripts executable"
	@echo ""
	@echo "Testing:"
	@echo "  test       - Run test suite"
	@echo ""
	@echo "Usage Examples:"
	@echo "  python video_interpolator.py -i input.mp4 -o output.mp4 -m temporal -f 60"
	@echo "  ./interpolate.sh -i input.mp4 -o output.mp4 -m optical_flow -f 120"
	@echo ""
	@echo "Cleaning:"
	@echo "  clean      - Remove test files"
	@echo ""
	@echo "For more information, see README.md"

# Make scripts executable
install:
	@echo "Making scripts executable..."
	chmod +x video_interpolator.py
	chmod +x interpolate.sh
	chmod +x test_interpolation.py
	@echo "✓ Scripts are now executable"

# Run test suite
test:
	@echo "Running test suite..."
	python test_interpolation.py

# Clean up test files
clean:
	@echo "Cleaning up test files..."
	rm -f test_input.mp4
	rm -f test_output_*.mp4
	rm -rf temp_frames
	@echo "✓ Test files cleaned up"

# Quick test with frame duplication
test-fast:
	@echo "Running quick test with frame duplication..."
	python video_interpolator.py -i test_input.mp4 -o test_output_fast.mp4 -m frame_duplication -f 60

# Show help for the main tool
help-tool:
	python video_interpolator.py --help

# Show help for the shell wrapper
help-shell:
	./interpolate.sh --help 