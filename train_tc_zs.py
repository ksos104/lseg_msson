from modules_tc.lseg_module_zs import LSegModuleZS
from utils import do_training, get_default_argument_parser

if __name__ == "__main__":
    parser = LSegModuleZS.add_model_specific_args(get_default_argument_parser())
    args = parser.parse_args()
    do_training(args, LSegModuleZS)
