"""Download and verify the original public pilot adapter. No credentials needed."""
import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL='https://github.com/elevate007/monash-faq-research/releases/download/v0.1.0/monash_trained_adapter.zip'
SHA256='e69717a1a00ae0b613eb9d60a70f2f0b43eb2d14ec0dcd57c7c5c1580df12061'
ROOT=Path(__file__).resolve().parents[1]

def install(archive,output):
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest!=SHA256:raise ValueError('Adapter archive checksum does not match the verified pilot export')
    output=output.resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty; existing models are not overwritten')
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive integrity failed')
        for member in z.infolist():
            parts=PurePosixPath(member.filename).parts
            if not parts or parts[0]!='monash_qa_adapter':continue
            if '..' in parts or PurePosixPath(member.filename).is_absolute() or '\\' in member.filename:
                raise ValueError('Unsafe archive path')
            if member.is_dir():continue
            target=output.joinpath(*parts[1:]).resolve()
            if not target.is_relative_to(output):raise ValueError('Unsafe extraction target')
            if (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive links are not supported')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as source,target.open('wb') as dest:shutil.copyfileobj(source,dest)
    if not (output/'adapter_model.safetensors').is_file():raise ValueError('Adapter weights missing in archive')
    print('Verified original pilot adapter installed:',output)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'models/monash_qa_adapter')
    args=parser.parse_args()
    if args.archive:install(args.archive,args.output)
    else:
        with tempfile.TemporaryDirectory(prefix='monash_adapter_') as folder:
            archive=Path(folder)/'adapter.zip'
            with urllib.request.urlopen(URL,timeout=60) as source,archive.open('wb') as dest:shutil.copyfileobj(source,dest)
            install(archive,args.output)
