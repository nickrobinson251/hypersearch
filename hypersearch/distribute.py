def distribute_compute(*args)
    LOGGER = logging.getLogger(__name__)
    LOGGER.addHandler(logging.StreamHandler)

    cluster = args.cluster
    CLIENT = Client()

    digits = load_digits()
    X, y = digits.data, digits.target
    model = RandomForestClassifier(n_estimators=20)

    args = parse_args()
    params = parse_params("../examples/params.yaml", args.method)

    with joblib.parallel_backend('dask.distributed'):
        result = search(model, X, y, params, method=args.method)
        joblib.dump(result, args.filepath)
