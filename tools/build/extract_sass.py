#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from flash_helpers.kernel_configs import (
    transform_kernel_name_to_short_form,
)


def run_command(cmd, check=True):
    """Run a shell command and return the output.

    Args:
        cmd: Command to run
        check: Whether to raise an exception on non-zero exit code

    Returns:
        Command output as string

    Raises:
        subprocess.CalledProcessError: If the command fails and check=True
    """
    try:
        process = subprocess.run(
            cmd, shell=True, text=True, capture_output=True, check=check
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Standard output: {e.stdout}")
        print(f"Standard error: {e.stderr}")
        if check:
            raise
        return e.stdout


def sanitize_filename(name):
    """Make the filename filesystem friendly."""
    # Remove spaces within parentheses
    name = re.sub(
        r"\(\s*([^)]+)\s*\)",
        lambda m: "(" + m.group(1).replace(" ", "") + ")",
        name,
    )

    # Replace characters that are problematic in filenames
    name = re.sub(r'[<>:"/\\|?*]', "_", name)

    # Replace remaining spaces with underscores
    name = name.replace(" ", "_")

    return name


def process_kernel(
    idx, func_name, cubin_path, output_dir, args, log_fn, input_file_path=None
):
    """Process a single kernel."""
    try:
        kern_name = transform_kernel_name_to_short_form(func_name)

        if args.list:
            print(f"{idx:5} {kern_name}")
            return

        log_fn(f"Processing kernel index {idx}: {func_name}")

        # Get the short form and sanitize for filename
        short_form = kern_name
        file_extension = ".dot" if args.cfg else ".asm"
        if args.fun is not None and input_file_path:
            # When -fun is specified, use input filename without extension
            input_without_ext = os.path.splitext(
                os.path.basename(input_file_path)
            )[0]
            filename = os.path.join(
                output_dir, f"{input_without_ext}{file_extension}"
            )
        elif args.name:
            filename = os.path.join(
                output_dir, f"{args.name}_{idx}{file_extension}"
            )
        else:
            filename = os.path.join(
                output_dir, f"{sanitize_filename(short_form)}{file_extension}"
            )

        # Run nvdisasm and save to file
        output_type = "control flow graph" if args.cfg else "disassembly"
        log_fn(f"Generating {output_type} for kernel {idx} to {filename}...")
        disasm_flag = "-cfg" if args.cfg else "-c"

        # Build nvdisasm command with optional additional arguments
        nvdisasm_cmd = f"nvdisasm {cubin_path} -fun {idx} {disasm_flag}"
        if args.nvdisasm_args:
            nvdisasm_cmd += f" {args.nvdisasm_args}"
        nvdisasm_cmd += " | cu++filt"

        disasm_output = run_command(nvdisasm_cmd)

        # Process output to remove offsets if requested (only for regular assembly, not CFG)
        if args.no_offsets and not args.cfg:
            # Remove lines that contain /*[alphanum]*/ and ensure exactly 8 characters before instruction
            processed_lines = []
            for line in disasm_output.splitlines():
                # Match lines with offset prefixes like /*0340*/
                match = re.match(r"^\s*/\*[0-9a-fA-F]+\*/\s*(.*)", line)
                if match:
                    # Extract the instruction part
                    instruction_part = match.group(1).strip()

                    # Check if there's a predicate (@P or @!P followed by digits or T)
                    predicate_match = re.match(
                        r"^(@!?P(?:\d+|T))\s+(.*)", instruction_part
                    )
                    if predicate_match:
                        predicate = predicate_match.group(1)
                        instruction = predicate_match.group(2)
                        # Calculate spaces needed so instruction starts at 8th character
                        predicate_length = (
                            len(predicate) + 1
                        )  # +1 for the space after predicate
                        spaces_needed = 8 - predicate_length
                        processed_line = (
                            " " * spaces_needed + predicate + " " + instruction
                        )
                    else:
                        # No predicate, use 8 spaces
                        processed_line = "        " + instruction_part
                    processed_lines.append(processed_line)
                else:
                    processed_lines.append(line)
            disasm_output = "\n".join(processed_lines)

        with open(filename, "w") as f:
            f.write(disasm_output)

        log_fn(f"Successfully saved disassembly to {filename}")

    except Exception as e:
        log_fn(f"Error processing kernel {idx}: {e}")


def get_kernels_from_cubin(cubin_path, log_fn):
    """Extract kernel information from a cubin file."""
    log_fn(f"Reading symbol table from {cubin_path}...")
    readelf_output = run_command(
        f"readelf -Ws {cubin_path} | grep FUNC | cu++filt"
    )

    kernels = []
    for line in readelf_output.splitlines():
        if not line.strip():
            continue

        if "void" not in line and "flash" not in line:
            continue

        # Parse the line to extract the relevant information
        match = re.match(
            r"\s*(\d+):\s+[0-9a-f]+\s+\d+\s+\w+\s+\w+\s+\w+\s+(?:\[<other>: \d+\])?\s+\d+\s+(void\s+.*)",
            line,
        )

        if not match:
            continue

        idx = match.group(1)
        func_name = match.group(2)

        kernels.append((idx, func_name))

    return kernels


def check_cuda_tools():
    """Check if required CUDA tools are available."""
    required_tools = ["cuobjdump", "nvdisasm", "readelf", "cu++filt"]
    missing_tools = []

    for tool in required_tools:
        try:
            result = subprocess.run(
                f"which {tool}", shell=True, text=True, capture_output=True
            )
            if result.returncode != 0:
                missing_tools.append(tool)
        except Exception:
            missing_tools.append(tool)

    if missing_tools:
        print(
            f"Error: The following required tools are missing: {', '.join(missing_tools)}"
        )
        print("Please make sure CUDA toolkit is installed and in your PATH.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process .so or .cubin files to extract kernel information."
    )
    parser.add_argument("input_file", help="Path to .so or .cubin file")
    parser.add_argument(
        "-cubin", help="Path to .cubin file (optional when input is .so)"
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List kernel indices and function names without disassembling",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-fun", type=int, help="Process only the specified function index"
    )
    parser.add_argument(
        "-name",
        help="Base name for output files (will be saved as name_idx.asm or name_idx.dot)",
    )
    parser.add_argument(
        "-no_offsets",
        action="store_true",
        help="Remove offset prefixes like /*0340*/ from assembly output and replace with 4 spaces",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for assembly/dot files (default: current directory)",
    )
    parser.add_argument(
        "-cfg",
        action="store_true",
        help="Generate control flow graph using nvdisasm -cfg instead of regular disassembly",
    )
    parser.add_argument(
        "--nvdisasm-args",
        help="Additional arguments to pass to nvdisasm (e.g., '--nvdisasm-args=\"-hex -ndf\"')",
    )

    args = parser.parse_args()

    # Check if required tools are available
    check_cuda_tools()

    # Function to print logs only if not in list mode
    def log(message):
        if not args.list:
            print(message)

    # Function for verbose logs
    def verbose_log(message):
        if args.verbose:
            print(f"VERBOSE: {message}")

    # Convert input file to absolute path
    input_file = os.path.abspath(args.input_file)

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Determine if we're processing a .so or .cubin file
    input_path = Path(input_file)
    is_so = input_path.suffix.lower() == ".so"

    # Set output directory
    output_dir = os.path.abspath(args.output) if args.output else os.getcwd()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log(f"Created output directory: {output_dir}")

    # If .so file and no cubin provided, attempt to extract cubin name
    if is_so and not args.cubin:
        log("No cubin file provided, attempting to find automatically...")

        try:
            verbose_log(f"Running: cuobjdump -lelf {input_file}")
            cuobjdump_output = run_command(f"cuobjdump -lelf {input_file}")
        except subprocess.CalledProcessError:
            print("Failed to extract ELF information from the .so file.")
            print(
                "Try running 'cuobjdump -lelf your_file.so' manually to diagnose the issue."
            )
            print(
                "If you know the cubin name, specify it with the -cubin argument."
            )

            # Try to get more information about the file
            print("\nFile information:")
            try:
                file_info = run_command(f"file {input_file}", check=False)
                print(file_info)
            except Exception as e:
                print(f"Could not get file information: {e}")

            # Try to list all embedded files without extracting
            try:
                print("\nAttempting to list all sections in the file:")
                sections = run_command(
                    f"cuobjdump -all {input_file}", check=False
                )
                print(sections)
            except Exception as e:
                print(f"Could not list sections: {e}")

            sys.exit(1)

        # Parse the output to find cubin files
        cubin_files = []
        for line in cuobjdump_output.splitlines():
            match = re.search(r"ELF file\s+\d+:\s+(.*\.cubin)", line)
            if match:
                cubin_files.append(match.group(1))

        if not cubin_files:
            print("Error: No cubin files found in the .so file.")
            print("cuobjdump output:")
            print(cuobjdump_output)

            # Try alternative command
            print("\nTrying alternative command to list sections...")
            try:
                alt_output = run_command(
                    f"cuobjdump -all {input_file}", check=False
                )
                print(alt_output)
            except Exception as e:
                print(f"Alternative command failed: {e}")

            sys.exit(1)

        if len(cubin_files) > 1:
            print("Error: Multiple cubin files found in the .so file:")
            print("\n".join(cubin_files))
            print("Please specify which one to use with the -cubin argument.")
            sys.exit(1)

        # We have exactly one cubin file - this is just the name, not the path
        cubin_name = cubin_files[0]
        log(f"Found cubin file: {cubin_name}")
        cubin_path = cubin_name
    else:
        cubin_path = args.cubin if is_so and args.cubin else input_file

    # If .so file, run cuobjdump in a temporary directory
    temp_dir = None

    try:
        if is_so:
            log("Extracting .cubin from .so file in temporary directory...")
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            verbose_log(f"Created temporary directory: {temp_dir}")

            # If manually specified cubin, convert to absolute path
            if args.cubin:
                cubin_path = os.path.abspath(args.cubin)

            # Change to the temporary directory and run cuobjdump
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            verbose_log(f"Changed directory to: {temp_dir}")

            # Run cuobjdump to extract the cubin
            extract_cmd = f"cuobjdump -xelf {cubin_path} {input_file}"
            log(f"Running: {extract_cmd}")
            try:
                run_command(extract_cmd)
            except subprocess.CalledProcessError as e:
                log(f"Error running cuobjdump: {e}")
                log("Trying with just the cubin name instead of the path...")
                basename_cmd = f"cuobjdump -xelf {os.path.basename(cubin_path)} {input_file}"
                verbose_log(f"Trying alternative command: {basename_cmd}")
                try:
                    run_command(basename_cmd)
                    cubin_path = os.path.basename(cubin_path)
                except subprocess.CalledProcessError:
                    print(
                        "Could not extract cubin file. Trying other extraction methods..."
                    )
                    # Try extracting all sections as a last resort
                    try:
                        verbose_log("Trying to extract all sections")
                        run_command(f"cuobjdump -xall {input_file}")

                        # Look for cubin files after extraction
                        cubin_files = list(Path(temp_dir).glob("*.cubin"))
                        if cubin_files:
                            cubin_path = str(cubin_files[0])
                            print(
                                f"Found cubin after extracting all sections: {cubin_path}"
                            )
                        else:
                            raise FileNotFoundError(
                                "No cubin files found after extraction"
                            )
                    except Exception as e:
                        print(f"All extraction methods failed: {e}")
                        print("Cannot continue without a valid cubin file.")
                        sys.exit(1)

            # Update cubin_path to the extracted file in the temp directory
            extracted_cubin = os.path.join(temp_dir, cubin_path)

            # Verify the extracted file exists
            if not os.path.exists(extracted_cubin):
                log(
                    f"Warning: Expected extracted file {extracted_cubin} not found"
                )
                # List files in temp dir
                log("Files in temporary directory:")
                dir_listing = run_command("ls -la")
                log(dir_listing)

                # Look for any .cubin files in the directory
                cubin_files = list(Path(temp_dir).glob("*.cubin"))
                if cubin_files:
                    extracted_cubin = str(cubin_files[0])
                    log(f"Found alternative .cubin file: {extracted_cubin}")
                else:
                    # List files in temp dir before exiting
                    log("Files in temporary directory:")
                    dir_listing = run_command("ls -la")
                    log(dir_listing)

                    raise FileNotFoundError(
                        f"Could not find any .cubin files in {temp_dir}"
                    )

            cubin_path = extracted_cubin
            verbose_log(f"Using cubin file at: {cubin_path}")

            # Get kernels and process them
            try:
                kernels = get_kernels_from_cubin(cubin_path, log)
                for idx, func_name in kernels:
                    # Skip if fun is specified and doesn't match the current index
                    if args.fun is not None and int(idx) != args.fun:
                        continue
                    process_kernel(
                        idx,
                        func_name,
                        cubin_path,
                        output_dir,
                        args,
                        log,
                        input_file,
                    )
            except Exception as e:
                print(f"Error processing kernels: {e}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()

            # Return to original directory
            os.chdir(original_dir)
        else:
            # For .cubin file, just process directly
            try:
                kernels = get_kernels_from_cubin(cubin_path, log)
                for idx, func_name in kernels:
                    # Skip if fun is specified and doesn't match the current index
                    if args.fun is not None and int(idx) != args.fun:
                        continue
                    process_kernel(
                        idx,
                        func_name,
                        cubin_path,
                        output_dir,
                        args,
                        log,
                        input_file,
                    )
            except Exception as e:
                print(f"Error processing kernels: {e}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()

        # If we're just listing, exit now
        if args.list:
            sys.exit(0)

    finally:
        # Clean up temporary directory if it exists
        if temp_dir and os.path.exists(temp_dir):
            log(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
