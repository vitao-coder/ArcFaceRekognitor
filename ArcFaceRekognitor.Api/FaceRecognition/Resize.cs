using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ArcFaceRekognitor.Api.FaceRecognition
{
    public partial class SCRFD
    {

        private static Image<Rgb24> Resize(Image<Rgb24> image, Size size)
        {
            image.Mutate(x => x
                .AutoOrient()
                .Resize(new ResizeOptions
                {
                    Mode = ResizeMode.BoxPad,
                    Position = AnchorPositionMode.TopLeft,
                    Size = size
                })
                .BackgroundColor(Color.Black));
            return image;
        }

    }
}
