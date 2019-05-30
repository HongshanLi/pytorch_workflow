

"""Arguments:
    num_workers
    batch_size
    start_epoch: which epoch to start
    epochs: num of epochs to train
    lr: learning rate
    log_dir: directory to write log files
    ckp_dir: directory to save checkpoints
    print_step: num of step to print progess

"""
def main():
    train_dataset = MyDataset('train')
    val_dataset = MyDataset('validate')

    train_loader = DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers)

    val_loader = DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)


    model = MyModel()
    criterion = Criterion()
    
    optimizer = Optimizer()

    trainer = Trainer(train_loader=train_loader, 
            model=model, criterion=criterion, 
            optimizer=optimizer, args)

    validator = Validator(val_loader, model, criterion, args)

    if args.estimate:
        # print some stuff to show model size
        # GPU memories need to hold the model
        # amount of data 
        # print total training steps for one epoch


        return

    if args.evaluate:
        validator()
        return

    if args.train:
        for epoch in range(args.start_epoch, args.epochs):
            trainer(epoch)
            validator()
            validator.is_best(epoch)






