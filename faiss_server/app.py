import faiss_service


app = faiss_service.create_app()


if __name__ == '__main__':
	app.run("0.0.0.0", port=9090)