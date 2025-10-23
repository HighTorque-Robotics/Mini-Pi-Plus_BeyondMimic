import argparse
import pandas as pd
import os
from pathlib import Path
from rich import print


def cut_motion_csv(input_csv, output_csv, start_frame, end_frame, remove_frame_column=False, z_offset=0.0, decimal_places=6):
    """
    Cut motion data CSV from start_frame to end_frame (inclusive)
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file  
        start_frame: Starting frame number (inclusive)
        end_frame: Ending frame number (inclusive)
        remove_frame_column: Whether to remove 'frame' column from output
        z_offset: Offset to add to root_pos_z values
        decimal_places: Number of decimal places to keep for numeric data
    """
    
    # Load CSV
    df = pd.read_csv(input_csv)
    
    print(f"Original data: {len(df)} frames")
    print(f"Frame range: {df['frame'].min()} to {df['frame'].max()}")
    
    # Validate frame range
    if start_frame < df['frame'].min() or start_frame > df['frame'].max():
        raise ValueError(f"Start frame {start_frame} is out of range [{df['frame'].min()}, {df['frame'].max()}]")
    
    if end_frame < df['frame'].min() or end_frame > df['frame'].max():
        raise ValueError(f"End frame {end_frame} is out of range [{df['frame'].min()}, {df['frame'].max()}]")
    
    if start_frame > end_frame:
        raise ValueError(f"Start frame {start_frame} must be <= end frame {end_frame}")
    
    # Cut the data
    cut_df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)].copy()
    
    # Reset frame numbers to start from 0
    cut_df['frame'] = cut_df['frame'] - start_frame
    
    # Reset time to start from 0
    if 'time' in cut_df.columns:
        fps = cut_df['fps'].iloc[0] if 'fps' in cut_df.columns else 30
        cut_df['time'] = cut_df['frame'] / fps
    
    print(f"Cut data: {len(cut_df)} frames (frames {start_frame} to {end_frame})")
    
    # Step 2: Add z_offset to root_pos_z column
    if z_offset != 0.0 and 'root pos z' in cut_df.columns:
        cut_df['root pos z'] += z_offset
        print(f"Added {z_offset}m offset to root_pos_z")
    
    # Step 3: Round numeric columns to specified decimal places
    numeric_columns = cut_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    for col in numeric_columns:
        if col != 'frame':  # Don't round frame numbers
            cut_df[col] = cut_df[col].round(decimal_places)
    print(f"Rounded numeric data to {decimal_places} decimal places")
    
    # Step 1: Remove frame column if requested (done last so frame is available for processing)
    if remove_frame_column and 'frame' in cut_df.columns:
        cut_df = cut_df.drop('frame', axis=1)
        print("Removed 'frame' column")
    
    # Save cut data
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cut_df.to_csv(output_csv, index=False)
    print(f"Saved processed motion to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Cut motion data CSV by frame range")
    
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--output_csv", 
        required=True,
        help="Path to output CSV file"
    )
    
    parser.add_argument(
        "--start_frame",
        type=int,
        required=True,
        help="Start frame number (inclusive)"
    )
    
    parser.add_argument(
        "--end_frame", 
        type=int,
        required=True,
        help="End frame number (inclusive)"
    )
    
    # Optional: batch processing
    parser.add_argument(
        "--input_folder",
        help="Folder containing CSV files for batch processing"
    )
    
    parser.add_argument(
        "--output_folder",
        help="Output folder for batch processing" 
    )
    
    parser.add_argument(
        "--suffix",
        default="_cut",
        help="Suffix to add to output filenames in batch mode (default: '_cut')"
    )
    
    parser.add_argument(
        "--remove_frame_column",
        action="store_true",
        default=False,
        help="Remove the 'frame' column from output CSV"
    )
    
    parser.add_argument(
        "--z_offset",
        type=float,
        default=0.0,
        help="Add offset to root_pos_z values (default: 0.0, example: 0.065)"
    )
    
    parser.add_argument(
        "--decimal_places",
        type=int,
        default=6,
        help="Number of decimal places to keep for numeric data (default: 6)"
    )

    args = parser.parse_args()
    
    if args.input_folder and args.output_folder:
        # Batch processing mode
        input_folder = Path(args.input_folder)
        output_folder = Path(args.output_folder)
        
        print(f"Batch processing CSV files in: {input_folder}")
        
        csv_files = list(input_folder.glob("*.csv"))
        if not csv_files:
            print("No CSV files found in input folder")
            return
            
        for csv_file in csv_files:
            output_name = csv_file.stem + args.suffix + ".csv"
            output_path = output_folder / output_name
            
            try:
                print(f"\nProcessing: {csv_file.name}")
                cut_motion_csv(
                    str(csv_file), 
                    str(output_path), 
                    args.start_frame, 
                    args.end_frame,
                    args.remove_frame_column,
                    args.z_offset,
                    args.decimal_places
                )
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
                
        print(f"\nBatch processing complete. Output saved to: {output_folder}")
        
    else:
        # Single file mode
        if not args.input_csv or not args.output_csv:
            print("For single file mode, both --input_csv and --output_csv are required")
            return
            
        try:
            cut_motion_csv(
                args.input_csv, 
                args.output_csv, 
                args.start_frame, 
                args.end_frame,
                args.remove_frame_column,
                args.z_offset,
                args.decimal_places
            )
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()