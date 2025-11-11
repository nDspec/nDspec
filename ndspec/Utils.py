def parse_plot_axes(plot):
    """
    This function is used to parse the contents of a matplotlib plot object. 
    Given a plot object return by e.g. one of the nDspec fitter object, it 
    loops over all the axis objects in the plots, and prints information about 
    the collections (which contain data points, colors, linewidth etc) in the 
    plot object.
    """

    for i, ax in enumerate(plot.axes):
        print(f"\n--- Plot axis #{i+1} ---")
        for j, collection in enumerate(ax.collections):
            print(f"--- Collection {j} ---")
            print(f"Type: {type(collection)}")
        
            # For nDspec plots, offsets are not relevant
            #if hasattr(collection, 'get_offsets'):
            #    offsets = collection.get_offsets()
            #    print(f"Number of offsets (data points): {len(offsets)}")
            #    print(f"First 3 offsets (x, y):\n{offsets[:3]}")
        
            # Print the number of segments (for errorbars, lines, etc.)
            if hasattr(collection, 'get_segments'):
                segments = collection.get_segments()
                print(f"Number of segments: {len(segments)}")
                print(f"First segment:\n{segments[0]}")
        
            # Print other useful attributes
            if hasattr(collection, 'get_array'):
                array = collection.get_array()
                print(f"Array data (e.g., colors, sizes): {array}")
        
            if hasattr(collection, 'get_linewidths'):
                linewidths = collection.get_linewidths()
                print(f"Line widths: {linewidths}")
        
            if hasattr(collection, 'get_colors'):
                colors = collection.get_colors()
                print(f"Colors: {colors}") 
                
def parse_plot_lines(plot):
    """
    This function is used to parse the contents of a matplotlib plot object. 
    Given a plot object return by e.g. one of the nDspec fitter object, it 
    loops over all the axis objects in the plots, and prints information about 
    the lines (which in nDspec contain e.g. the model info) stored in the plot. 
    """
    for j, ax in enumerate(plot.axes):
        print(f"\n--- Plot axis #{j+1} ---")
        for i, line in enumerate(ax.get_lines()):
            print(f"--- Line {i} ---")
            print(f"Label: {line.get_label()}")
            print(f"Color: {line.get_color()}")
            print(f"Linestyle: {line.get_linestyle()}")
            print(f"Linewidth: {line.get_linewidth()}")
            print(f"X data length: {len(line.get_xdata())}")
            print(f"Y data length: {len(line.get_ydata())}")
            print(f"First 3 (x, y) points: {list(zip(line.get_xdata()[:3], line.get_ydata()[:3]))}")
