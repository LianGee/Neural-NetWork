#-------------------------------------------------------------------------------
# Name:        3LayerBp
# Purpose:
#
# Author:      LianGee
#
# Created:     15/12/2015
# Copyright:   (c) LianGee 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def nonlin(x, deriv = False):
    if (deriv == True):
        return 1
    else:
        return 0
