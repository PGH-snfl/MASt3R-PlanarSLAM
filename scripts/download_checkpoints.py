"""Download upstream MASt3R weights; existing files are left intact."""
from pathlib import Path
import shutil
import urllib.request

BASE = 'https://download.europe.naverlabs.com/ComputerVision/MASt3R/'
FILES = (
    'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
    'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth',
    'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl',
)


def main():
    target = Path(__file__).resolve().parents[1] / 'checkpoints'
    target.mkdir(exist_ok=True)
    for name in FILES:
        path = target / name
        if path.exists():
            print(f'Exists, skipping: {path}')
            continue
        temporary = path.with_suffix(path.suffix + '.part')
        print(f'Downloading {name}', flush=True)
        try:
            with urllib.request.urlopen(BASE + name, timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
