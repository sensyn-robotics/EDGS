#!/usr/bin/env python3
import struct
import sys
import argparse

def read_ply_header(file):
    header_lines = []
    vertex_count = 0
    properties = []
    
    while True:
        line = file.readline().decode('utf-8').strip()
        header_lines.append(line)
        
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[2])
        elif line.startswith('property'):
            parts = line.split()
            prop_type = parts[1]
            prop_name = parts[2]
            properties.append((prop_type, prop_name))
        elif line == 'end_header':
            break
    
    return header_lines, vertex_count, properties

def rescale_ply(input_file, output_file, scale_x=1.0, scale_y=1.0, scale_z=1.0, scale_uniform=None):
    if scale_uniform is not None:
        scale_x = scale_y = scale_z = scale_uniform
    
    import math
    # For Gaussian Splatting, scales are in log space
    log_scale_x = math.log(scale_x)
    log_scale_y = math.log(scale_y)
    log_scale_z = math.log(scale_z)
    
    with open(input_file, 'rb') as f_in:
        header_lines, vertex_count, properties = read_ply_header(f_in)
        
        vertex_data = []
        bytes_per_float = 4
        bytes_per_vertex = len(properties) * bytes_per_float
        
        for i in range(vertex_count):
            vertex_bytes = f_in.read(bytes_per_vertex)
            vertex_floats = struct.unpack(f'<{len(properties)}f', vertex_bytes)
            
            scaled_vertex = []
            for j, (prop_type, prop_name) in enumerate(properties):
                value = vertex_floats[j]
                
                if prop_name == 'x':
                    value *= scale_x
                elif prop_name == 'y':
                    value *= scale_y
                elif prop_name == 'z':
                    value *= scale_z
                elif prop_name in ['scale_0', 'scale_1', 'scale_2']:
                    # Gaussian scales are stored in log space
                    # To scale them, we add the log of the scale factor
                    if prop_name == 'scale_0':
                        value += log_scale_x
                    elif prop_name == 'scale_1':
                        value += log_scale_y
                    elif prop_name == 'scale_2':
                        value += log_scale_z
                
                scaled_vertex.append(value)
            
            vertex_data.append(scaled_vertex)
        
        with open(output_file, 'wb') as f_out:
            for line in header_lines:
                f_out.write((line + '\n').encode('utf-8'))
            
            for vertex in vertex_data:
                vertex_bytes = struct.pack(f'<{len(properties)}f', *vertex)
                f_out.write(vertex_bytes)
    
    print(f"Rescaled PLY file saved to: {output_file}")
    print(f"Applied scaling: X={scale_x}, Y={scale_y}, Z={scale_z}")
    print(f"Processed {vertex_count} vertices")

def main():
    parser = argparse.ArgumentParser(description='Rescale a PLY point cloud file')
    parser.add_argument('input', help='Input PLY file')
    parser.add_argument('output', help='Output PLY file')
    parser.add_argument('--scale', type=float, help='Uniform scale factor for all axes')
    parser.add_argument('--scale-x', type=float, default=1.0, help='Scale factor for X axis')
    parser.add_argument('--scale-y', type=float, default=1.0, help='Scale factor for Y axis')
    parser.add_argument('--scale-z', type=float, default=1.0, help='Scale factor for Z axis')
    
    args = parser.parse_args()
    
    rescale_ply(
        args.input,
        args.output,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        scale_z=args.scale_z,
        scale_uniform=args.scale
    )

if __name__ == '__main__':
    main()