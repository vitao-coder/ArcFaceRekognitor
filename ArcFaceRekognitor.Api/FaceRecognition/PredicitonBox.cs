namespace ArcFaceRekognitor.Api.FaceRecognition
{
    public class PredictionBox
    {
        private readonly float score;
        private readonly float boxLeft;
        private readonly float boxRight;
        private readonly float boxBottom;
        private readonly float boxTop;
        private readonly float[] landmark;

        public PredictionBox(float score, float boxLeft, float boxTop, float boxRight, float boxBottom, float[] landmark)
        {
            this.score = score;
            this.boxLeft = boxLeft;
            this.boxRight = boxRight;
            this.boxBottom = boxBottom;
            this.boxTop = boxTop;
            this.landmark = landmark;
        }

        public float Score => score;

        public float BoxLeft => boxLeft;

        public float BoxRight => boxRight;

        public float BoxBottom => boxBottom;

        public float BoxTop => boxTop;

        public float[] Landmark => landmark;
    }
}
