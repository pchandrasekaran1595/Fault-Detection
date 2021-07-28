"""
    CLI Application
"""

import sys
from time import time

import utils as u
import Models
from Snapshot import capture_snapshot
from MakeData import make_data
from Train import trainer
from RTApp import realtime

# ******************************************************************************************************************** #

def app():    
    args_1 = "--num-samples"
    args_2 = "--embed"
    args_3 = "--epochs"
    args_4 = "--id"

    args_5 = "--lower"
    args_6 = "--upper"
    args_7 = "--early"

    # CLI Argument Handling
    if args_1 in sys.argv:
        u.num_samples = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        u.embed_layer_size = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        u.epochs = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        u.device_id = int(sys.argv[sys.argv.index(args_4) + 1])
    
    if args_5 in sys.argv:
        u.lower_bound_confidence = float(sys.argv[sys.argv.index(args_5) + 1])
    if args_6 in sys.argv:
        u.upper_bound_confidence = float(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv:
        u.early_stopping_step = int(sys.argv[sys.argv.index(args_7) + 1])
    
    while True:
        u.breaker()
        ch = input("1. Add Object\n2. Retrain\n3. Application\n4. Exit\n\nEnter Choice : ")

        if ch == "1":
            u.breaker()
            part_name = input("Enter part name : ")
            capture_snapshot(device_id=u.device_id, part_name=part_name, roi_extractor=Models.roi_extractor)

            u.breaker()
            u.myprint("Generating Feature Vector Data ...", "green")
            start_time = time()
            make_data(part_name=part_name, cls="Positive", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            make_data(part_name=part_name, cls="Negative", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            u.myprint("\nTime Taken [{}] : {:.2f} minutes".format(2*u.num_samples, (time()-start_time)/60), "green")

            model, batch_size, lr, wd = Models.build_siamese_model(embed=u.embed_layer_size)
            trainer(part_name=part_name, model=model, epochs=u.epochs, lr=lr, wd=wd, batch_size=batch_size, early_stopping=u.early_stopping_step, fea_extractor=Models.fea_extractor)
            realtime(device_id=u.device_id, part_name=part_name, model=model, save=False, fea_extractor=Models.fea_extractor)
        
        elif ch == "2":
            u.breaker()
            part_name = input("Enter part name : ")

            u.breaker()
            u.myprint("Generating Feature Vector Data ...", "green")
            start_time = time()
            make_data(part_name=part_name, cls="Positive", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            make_data(part_name=part_name, cls="Negative", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            u.myprint("\nTime Taken [{}] : {:.2f} minutes".format(2*u.num_samples, (time()-start_time)/60), "green")

            model, batch_size, lr, wd = Models.build_siamese_model(embed=u.embed_layer_size)
            trainer(part_name=part_name, model=model, epochs=u.epochs, lr=lr, wd=wd, batch_size=batch_size, early_stopping=u.early_stopping_step, fea_extractor=Models.fea_extractor)
        
        elif ch == "3":
            model, _, _, _ = Models.build_siamese_model(embed=u.embed_layer_size)
            u.breaker()
            part_name = input("Enter part name : ")
            realtime(device_id=u.device_id, part_name=part_name, model=model, save=False)

        elif ch == "4":
            break
            
        else:
            u.myprint("\nINVALID CHOICE !!!", "red")

# ******************************************************************************************************************** #
