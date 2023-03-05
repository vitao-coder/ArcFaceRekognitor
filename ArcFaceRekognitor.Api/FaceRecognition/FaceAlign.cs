using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ArcFaceRekognitor.Api.FaceRecognition
{
    public class FaceAlign
    {
        static public Image<Rgb24> Align(Image<Rgb24> image, float[] landmarks, int width, int height)
        {
            float dy = landmarks[3] - landmarks[1], dx = landmarks[2] - landmarks[0];
            double angle = Math.Atan2(dy, dx) * 180f / Math.PI;

            float[] eye_center = new float[] { (landmarks[0] + landmarks[2]) / 2, (landmarks[1] + landmarks[3]) / 2 };
            float[] lip_center = new float[] { (landmarks[6] + landmarks[8]) / 2, (landmarks[7] + landmarks[9]) / 2 };

            float dis = (float)Math.Sqrt(Math.Pow(eye_center[0] - lip_center[0], 2) + Math.Pow(eye_center[1] - lip_center[1], 2)) / 0.35f;

            int bottom = (int)Math.Round(dis * 0.65, MidpointRounding.AwayFromZero);
            int top = (int)Math.Round(dis - bottom, MidpointRounding.AwayFromZero);
            int left = (int)Math.Round(width * dis / height / 2, MidpointRounding.AwayFromZero);

            int[] center = new int[] { (int)Math.Round(eye_center[0], MidpointRounding.AwayFromZero), (int)Math.Round(eye_center[1], MidpointRounding.AwayFromZero) };

            int x1 = Math.Max(center[0] - 2 * bottom, 0), y1 = Math.Max(center[1] - 2 * bottom, 0);
            int x2 = Math.Min(center[0] + 2 * bottom, image.Width - 1), y2 = Math.Min(center[1] + 2 * bottom, image.Height - 1);
            image.Mutate(img => img.Crop(new Rectangle(x1, y1, x2 - x1 + 1, y2 - y1 + 1)));

            int i_size = 4 * bottom + 1;
            var newImage = new Image<Rgb24>(i_size, i_size, backgroundColor: Color.White);

            newImage.Mutate(img =>
            {
                img.DrawImage(image, location: new SixLabors.ImageSharp.Point(2 * bottom - (center[0] - x1), 2 * bottom - (center[1] - y1)), opacity: 1);
                img.Rotate((float)(0 - angle));
                img.Crop(new Rectangle(newImage.Width / 2 - left, newImage.Height / 2 - top, left * 2, top + bottom));
                img.Resize(width, height);
            });

            return newImage;

        }

        static public Mat Align(Mat image, float[] landmarks, int width, int height)
        {       
            float[,] std = { { 38.2946f, 51.6963f }, { 73.5318f, 51.5014f }, { 56.0252f, 71.7366f }, { 41.5493f, 92.3655f }, { 70.7299f, 92.2041f } };
            Mat S = new Mat(5, 2, MatType.CV_32FC1, std);

            Mat Q = Mat.Zeros(10, 4, MatType.CV_32FC1);
            Mat S1 = S.Reshape(0, 10);

            for (int i = 0; i < 5; i++)
            {
                Q.At<float>(i * 2 + 0, 0) = landmarks[i * 2];
                Q.At<float>(i * 2 + 0, 1) = landmarks[i * 2 + 1];
                Q.At<float>(i * 2 + 0, 2) = 1;
                Q.At<float>(i * 2 + 0, 3) = 0;

                Q.At<float>(i * 2 + 1, 0) = landmarks[i * 2 + 1];
                Q.At<float>(i * 2 + 1, 1) = -landmarks[i * 2];
                Q.At<float>(i * 2 + 1, 2) = 0;
                Q.At<float>(i * 2 + 1, 3) = 1;
            }

            Mat MM = (Q.T() * Q).Inv() * Q.T() * S1;

            Mat M = new Mat(2, 3, MatType.CV_32FC1, new float[,] { { MM.At<float>(0, 0), MM.At<float>(1, 0), MM.At<float>(2, 0) }, { -MM.At<float>(1, 0), MM.At<float>(0, 0), MM.At<float>(3, 0) } });

            OpenCvSharp.Mat dst = new Mat();
            Cv2.WarpAffine(image, dst, M, new OpenCvSharp.Size(width, height));
            S.Release();
            Q.Release();
            S1.Release();
            MM.Release();
            M.Release();

            return dst;
        }
    }
}
