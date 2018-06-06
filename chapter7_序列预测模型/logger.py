#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : logger.py
# @Time    : 18-3-14
# @Author  : J.W.

import logging as logger

logger.basicConfig(level=logger.DEBUG,
                   format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S -',
                   filename='log.log',
                   filemode='a')  # or 'w', default 'a'

console = logger.StreamHandler()
console.setLevel(logger.INFO)
formatter = logger.Formatter('%(asctime)s %(name)-6s: %(levelname)-6s %(message)s')
console.setFormatter(formatter)
logger.getLogger('').addHandler(console)

#
# logger.info("info test.")
# logger.debug("debug test.")
# logger.warning('waring test.')
#
# # 指定logger名称
# logger1 = logger.getLogger("logger1")
# logger1.info('logger1 info test.')
