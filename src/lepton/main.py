# Std modules
import argparse
import sys

# Package modules
from lepton import ESC
from lepton import Lepton
from lepton import Videowriter


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', help="Lepton camera port", 
                        type=int, default=0)
    parser.add_argument('-r', "--record", help="record data stream", 
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-n', "--name", help="name of saved video file", 
                        type=str, default="recording")
    parser.add_argument('-c', "--cmap", help="colormap used in viewer", 
                        default='black_hot', 
                        choices=['afmhot', 'arctic', 'black_hot', 'cividis', 
                                 'ironbow', 'inferno', 'magma','outdoor_alert',
                                 'plasma', 'rainbow', 'rainbow_hc', 'viridis',
                                 'white_hot'])
    parser.add_argument('-sf', "--scale-factor", 
                        help="the amount the captured image is scaled by",
                        type=int, default=3)
    parser.add_argument('-eq', "--equalize", 
                        help="apply histogram equalization to image", 
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-d', "--detect", help="if moving fronts are detected", 
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-m', "--multiframe", help="detection type", 
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-f', "--fps", help="target FPS of camera", 
                        type=int, default=None)
    parser.add_argument('-o', "--overlay", 
                        help=argparse.SUPPRESS, 
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug", 
                        help=argparse.SUPPRESS, 
                        action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    return args


def leprun(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args()
    
    if not args.fps is None and args.fps < 5:
        wstr="Target FPS set below 5 can result in erroneous video rendering."
        print(ESC.WARNING+'WARNING: '+wstr+ESC.ENDC, flush=True)

    lepton = Lepton(args.port, args.cmap, args.scale_factor, 
                    args.overlay, args.debug)
    if not args.record:
        _ = lepton.start_stream(fps=args.fps, 
                                detect_fronts=args.detect, 
                                multiframe=args.multiframe,
                                equalize=args.equalize)
    
    else:
        _ = lepton.start_record(fps=args.fps,
                                detect_fronts=args.detect,
                                multiframe=args.multiframe, 
                                equalize=args.equalize)
        writer = Videowriter(rec_name=args.name, dirpath=lepton.DIRPATH)
        _ = writer.make_video()
        