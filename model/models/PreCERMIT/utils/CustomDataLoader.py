class CustomDataLoader:
    def __init__(self, json_path, img_directory, dots_directory, **params_train):
        self.json_path = json_path
        self.img_path = img_directory
        self.pcl_path = dots_directory
        self.num_imgs = random.randint(1, 24)
        self.batch_gen = BatchGen(json_path=json_path, image_directory=img_directory, dots_directory=dots_directory)
        self.batch_size = params_train['batch_size']
        self.is_shuffle = params_train['shuffle']
        self._index = 0  # Индекс для итерации

    def shuffle(self):
        data = list(range(len(self.batch_gen)))
        if self.is_shuffle:
            random.shuffle(data)
        return data

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.batch_gen):
            self._index = 0

        indices = self.shuffle()
        batch_dots = []
        batch_images = []

        while len(batch_dots) < self.batch_size:
            if self._index >= len(self.batch_gen):
                self._index = 0

            idx = indices[self._index]
            self.batch_gen.num_imgs = self.num_imgs
            image, dots = self.batch_gen[idx]

            batch_dots.append(dots.unsqueeze(0))
            batch_images.append(image.unsqueeze(0))

            self._index += 1

        batch_dots_tensor = torch.cat(batch_dots, dim=0)
        batch_images_tensor = torch.cat(batch_images, dim=0)
        self.num_imgs = random.randint(1, 24)

        return batch_images_tensor, batch_dots_tensor
