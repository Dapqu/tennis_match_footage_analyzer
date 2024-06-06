def get_center(bbox):
    """
    Calculates the center coordinates of a bounding box.
    
    Parameters:
    - bbox: A tuple or list containing the coordinates of the bounding box (x1, y1, x2, y2)
    
    Returns:
    - A tuple containing the center coordinates (center_x, center_y)
    """

    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Calculations
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return (center_x, center_y)

def measure_distance(pt1, pt2):
    """
    Calculates the distance between two points.
    
    Parameters:
    - pt1: A tuple or list containing the coordinates of the first point (x1, y1)
    - pt2: A tuple or list containing the coordinates of the second point (x2, y2)
    
    Returns:
    - The Euclidean distance between the two points
    """

    # Calculate the difference in the x-coordinates
    delta_x = pt1[0] - pt2[0]
    # Calculate the difference in the y-coordinates
    delta_y = pt1[1] - pt2[1]
    # Calculate the distance using the Pythagorean Theorem
    distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
    
    return distance