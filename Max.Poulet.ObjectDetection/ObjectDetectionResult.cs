namespace Max.Poulet.ObjectDetection;
using Max.Poulet.ObjectDetection;
public record ObjectDetectionResult
{
    public byte[] ImageData { get; set; }
    public IList<BoundingBox> Box { get; set; }
} 
