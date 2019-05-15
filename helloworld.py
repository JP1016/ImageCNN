#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:24:00 2019

@author: jp
"""

import tensorflow as tf


# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, World!')
print(hello)
# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))