namespace Max.Poulet.ObjectDetection;
    public class BoundingBox
    {
        public string Label { get; set; }
        public float Confidence { get; set; }
        public BoundingBoxDimensions Dimensions { get; set; }
    }

    public class BoundingBoxDimensions
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
    }
