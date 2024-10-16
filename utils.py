def smooth_data(data, alpha=0.1):
    smoothed = [data[0]]
    for point in data[1:]:
        smoothed.append(alpha * point + (1 - alpha) * smoothed[-1])
    return smoothed