from argparse import ArgumentParser
from typing import Self


class ConfigParser:

    @classmethod
    def from_args(cls, args: ArgumentParser) -> Self:

        if not isinstance(args, tuple):
            args = args.parse_args()

