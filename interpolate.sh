#!/bin/bash

# Video Interpolation Tool Wrapper
# A simple shell script wrapper for the Python video interpolation tool

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Video Interpolation Tool"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE     Input video file (required)"
    echo "  -o, --output FILE    Output video file (required)"
    echo "  -m, --method METHOD  Interpolation method:"
    echo "                       frame_duplication (fastest)"
    echo "                       temporal (balanced)"
    echo "                       optical_flow (high quality)"
    echo "                       advanced (best quality)"
    echo "  -f, --fps FPS        Target frame rate (default: 60)"
    echo "  --info               Show video information only"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i input.mp4 -o output.mp4 -m temporal -f 60"
    echo "  $0 -i input.mp4 --info"
    echo "  $0 -i input.mp4 -o output.mp4 -m optical_flow -f 120"
    echo ""
}

# Function to check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        if ! command -v python &> /dev/null; then
            print_error "Python is not installed or not in PATH"
            exit 1
        else
            PYTHON_CMD="python"
        fi
    else
        PYTHON_CMD="python3"
    fi
    print_info "Using Python: $PYTHON_CMD"
}

# Function to check if the Python script exists
check_script() {
    if [ ! -f "video_interpolator.py" ]; then
        print_error "video_interpolator.py not found in current directory"
        exit 1
    fi
}

# Function to validate input file
validate_input() {
    if [ ! -f "$INPUT_FILE" ]; then
        print_error "Input file not found: $INPUT_FILE"
        exit 1
    fi
    
    # Check if it's a video file (basic check)
    if ! file "$INPUT_FILE" | grep -q "video\|Video"; then
        print_warning "File may not be a video: $INPUT_FILE"
    fi
}

# Function to create output directory
create_output_dir() {
    if [ -n "$OUTPUT_FILE" ]; then
        OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
        if [ -n "$OUTPUT_DIR" ] && [ ! -d "$OUTPUT_DIR" ]; then
            print_info "Creating output directory: $OUTPUT_DIR"
            mkdir -p "$OUTPUT_DIR"
        fi
    fi
}

# Main script
main() {
    # Default values
    INPUT_FILE=""
    OUTPUT_FILE=""
    METHOD="temporal"
    FPS=60
    INFO_ONLY=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--input)
                INPUT_FILE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -m|--method)
                METHOD="$2"
                shift 2
                ;;
            -f|--fps)
                FPS="$2"
                shift 2
                ;;
            --info)
                INFO_ONLY=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_python
    check_script
    
    # Validate input file
    if [ -z "$INPUT_FILE" ]; then
        print_error "Input file is required"
        show_usage
        exit 1
    fi
    
    validate_input
    
    # Handle info-only mode
    if [ "$INFO_ONLY" = true ]; then
        print_info "Getting video information..."
        $PYTHON_CMD video_interpolator.py -i "$INPUT_FILE" --info
        exit 0
    fi
    
    # Validate output file
    if [ -z "$OUTPUT_FILE" ]; then
        print_error "Output file is required when not using --info"
        show_usage
        exit 1
    fi
    
    # Create output directory if needed
    create_output_dir
    
    # Validate method
    case $METHOD in
        frame_duplication|temporal|optical_flow|advanced)
            ;;
        *)
            print_error "Invalid method: $METHOD"
            print_info "Valid methods: frame_duplication, temporal, optical_flow, advanced"
            exit 1
            ;;
    esac
    
    # Validate FPS
    if ! [[ "$FPS" =~ ^[0-9]+$ ]] || [ "$FPS" -lt 1 ]; then
        print_error "FPS must be a positive integer"
        exit 1
    fi
    
    # Show processing information
    print_info "Input file: $INPUT_FILE"
    print_info "Output file: $OUTPUT_FILE"
    print_info "Method: $METHOD"
    print_info "Target FPS: $FPS"
    echo ""
    
    # Run the Python script
    print_info "Starting video interpolation..."
    if $PYTHON_CMD video_interpolator.py -i "$INPUT_FILE" -o "$OUTPUT_FILE" -m "$METHOD" -f "$FPS"; then
        print_success "Video interpolation completed successfully!"
        print_info "Output saved to: $OUTPUT_FILE"
    else
        print_error "Video interpolation failed!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@" 