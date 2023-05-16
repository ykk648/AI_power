import cv2
import torch
from basicsr.utils import img2tensor, tensor2img  # pip install basicsr
from torchvision.transforms.functional import normalize
from .archs.gfpganv1_clean_arch import GFPGANv1Clean


class GFPGANer:

    def __init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'RestoreFormer':
            from .archs.restoreformer_arch import RestoreFormer
            self.gfpgan = RestoreFormer()
        elif arch == 'CodeFormer':
            from .archs.codeformer_arch import CodeFormer
            self.gfpgan = CodeFormer(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256'])

        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance_single_aligned_image(self, img, restore_former_weight=0.5):
        cropped_face = cv2.resize(img, (512, 512))

        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

        try:
            # output = self.gfpgan(cropped_face_t, return_rgb=False)[0]
            output = self.gfpgan(cropped_face_t, return_rgb=False, weight=restore_former_weight)[0]
            # convert to image
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        except RuntimeError as error:
            print(f'\tFailed inference for GFPGAN: {error}.')
            restored_face = cropped_face

        restored_face = restored_face.astype('uint8')
        return restored_face
