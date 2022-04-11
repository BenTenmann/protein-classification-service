## [1.4.1](https://github.com/BenTenmann/protein-classification-service/compare/deploy-1.4.0...deploy-1.4.1) (2022-04-11)


### Bug Fixes

* **cci:** login docker ([cc54485](https://github.com/BenTenmann/protein-classification-service/commit/cc54485cec010ccb37b3ef433b78cf0f45f55b17))

# [1.4.0](https://github.com/BenTenmann/protein-classification-service/compare/deploy-1.3.0...deploy-1.4.0) (2022-04-11)


### Bug Fixes

* **cci:** docker push command condition ([7839f8a](https://github.com/BenTenmann/protein-classification-service/commit/7839f8a654aef2e9c487fbc9acb19fe6e6ec000d))
* **cci:** setup remote docker ([5b2989d](https://github.com/BenTenmann/protein-classification-service/commit/5b2989d27887f9c5ebd66ca7ecf5a754a9c2be6d))
* **semrel:** add helm Chart to assets ([05d4a0e](https://github.com/BenTenmann/protein-classification-service/commit/05d4a0e11bdcd94bd220705c7a7b26f86a795d94))


### Features

* **helm:** add docker registry ([95fdead](https://github.com/BenTenmann/protein-classification-service/commit/95fdeadcc61c4dca5b70cd074c21741c701641e4))

# [1.3.0](https://github.com/BenTenmann/protein-classification-service/compare/deploy-1.2.0...deploy-1.3.0) (2022-04-11)


### Bug Fixes

* **docker:** pin werkzeug==2.0.3 to fix seldon dep ([3c4226c](https://github.com/BenTenmann/protein-classification-service/commit/3c4226cbdfaca01b0054173f139d292152f295b0))
* **docker:** remove env vars + remove model copy ([cfb0375](https://github.com/BenTenmann/protein-classification-service/commit/cfb0375766e142b596e8e463c38193eb1bebe3ee))
* **semrel:** add exec perm on yq ([95a7de6](https://github.com/BenTenmann/protein-classification-service/commit/95a7de653662e8f13f7ed892dea954303b33b44c))
* **semrel:** add helm to git + run script b4 semrel ([d14af7e](https://github.com/BenTenmann/protein-classification-service/commit/d14af7e5c6de3d4496daf8807a477d5ae21e63d2))


### Features

* **cci:** add docker build + push step ([efc62bb](https://github.com/BenTenmann/protein-classification-service/commit/efc62bbaf5a24e92bcf25bd54161fcb92749fab7))
* **helm:** add service helm chart ([59318b6](https://github.com/BenTenmann/protein-classification-service/commit/59318b6f2a78ea71cdfb9a4a2e3dc7438ccfc0fb))
* **semrel:** auto-bump image tags in helm ([41c6017](https://github.com/BenTenmann/protein-classification-service/commit/41c601713a7ac68b5d70b33bbfae09abebe1eedb))

# [1.2.0](https://github.com/BenTenmann/protein-classification-service/compare/deploy-1.1.0...deploy-1.2.0) (2022-02-18)


### Features

* add refs ([9193cf7](https://github.com/BenTenmann/protein-classification-service/commit/9193cf7aa6d4c2f39488c8f13f905a4235566e19))

# [1.1.0](https://github.com/BenTenmann/protein-classification-service/compare/deploy-1.0.3...deploy-1.1.0) (2022-02-17)


### Features

* add .get-tag + rm dl script ([8d36719](https://github.com/BenTenmann/protein-classification-service/commit/8d367196cc85236ff0cfcf12a57c35d25ea486d2))

## [1.0.3](https://github.com/BenTenmann/protein-classification-service/compare/deploy-1.0.2...deploy-1.0.3) (2022-02-17)


### Bug Fixes

* update cci-codecov ([21ab5a7](https://github.com/BenTenmann/protein-classification-service/commit/21ab5a710e3bb2451231bef0a4e3a1335940a9d6))
* update cci-codecov - 2 ([31f0d2c](https://github.com/BenTenmann/protein-classification-service/commit/31f0d2c9e7e1ceed27ab572578c0670ddb609c06))

## [1.0.2](https://github.com/BenTenmann/protein-classification-service/compare/deploy-1.0.1...deploy-1.0.2) (2022-02-17)


### Bug Fixes

* dev dep typo ([fdaaba1](https://github.com/BenTenmann/protein-classification-service/commit/fdaaba162f7f165c615a3f2540e4cff32f196930))

## [1.0.1](https://github.com/BenTenmann/protein-classification-service/compare/deploy-1.0.0...deploy-1.0.1) (2022-02-17)


### Bug Fixes

* dev dependencies ([2452c22](https://github.com/BenTenmann/protein-classification-service/commit/2452c220462205765e1a72f9074ba7e38bec547f))

# 1.0.0 (2022-02-17)


### Bug Fixes

* change mlp forward method ([c6a32aa](https://github.com/BenTenmann/protein-classification-service/commit/c6a32aaaa41641af3f5dff153ab9c614efa53f2e))
* dataset doc typing ([07b29ce](https://github.com/BenTenmann/protein-classification-service/commit/07b29ce86eb2a8e66d8ea64f8b1168cdd4aba414))
* device ([b280217](https://github.com/BenTenmann/protein-classification-service/commit/b28021720070f86e086d91adbe7ce8e753384b85))
* download script ([59ad3f4](https://github.com/BenTenmann/protein-classification-service/commit/59ad3f45801cab00efae3baf84ae42fae3c2e583))
* ens ([ee1ffda](https://github.com/BenTenmann/protein-classification-service/commit/ee1ffdac05b040f317977bb6dae9f0bbc7edd737))
* flatten ([939b5f2](https://github.com/BenTenmann/protein-classification-service/commit/939b5f29d2b54bc469dd2dc054958b56a5c612dd))
* import ([45564f4](https://github.com/BenTenmann/protein-classification-service/commit/45564f47aec7e1d288dabd564b9b5413832643ac))
* mlp flattening ([4524f11](https://github.com/BenTenmann/protein-classification-service/commit/4524f11ad9a955e2828d82ea9a07e7175afa0df5))
* remove html ([bb7ae90](https://github.com/BenTenmann/protein-classification-service/commit/bb7ae907e6e833fa13d419508fbbb5e6fe8c3daf))
* remove junk scripts ([2e88b83](https://github.com/BenTenmann/protein-classification-service/commit/2e88b83fbd29ac57c4047bd191832abbb673fc8a))
* remove jupyter notebooks ([0f13fd9](https://github.com/BenTenmann/protein-classification-service/commit/0f13fd91b0796e54a4980c36617e4dc0220df431))
* seldon service functional ([aa279d4](https://github.com/BenTenmann/protein-classification-service/commit/aa279d448aa6df2ae38215d4618fd4ae8f2d4961))
* wandb ([908b752](https://github.com/BenTenmann/protein-classification-service/commit/908b752ca7fcd72136dc7e96930af7625239a32c))


### Features

* add __init__ doc ([e8b0ffa](https://github.com/BenTenmann/protein-classification-service/commit/e8b0ffa29f819afff6623c52967072ab137c2cb0))
* add colab scripts ([ed909af](https://github.com/BenTenmann/protein-classification-service/commit/ed909affab61b1deaa8e5439315598d8b7728c7b))
* add dataset doc ([15a80a7](https://github.com/BenTenmann/protein-classification-service/commit/15a80a78f1674ed873a7b59fd7f7a30b894130fd))
* add dockerignore ([026c463](https://github.com/BenTenmann/protein-classification-service/commit/026c46318ddf4784bf49a6a4a18aa7125b16e95d))
* add download script ([a11148e](https://github.com/BenTenmann/protein-classification-service/commit/a11148eb421773c5ecd9ccbfe5b9002dc5e7decf))
* add one hot model ([24ef10e](https://github.com/BenTenmann/protein-classification-service/commit/24ef10ea6fdde80356d8e9ee3b2fa282845f96ee))
* add README ([7b4f216](https://github.com/BenTenmann/protein-classification-service/commit/7b4f2169df38e3701b81209a8bab3d60becd2496))
* add service ([66e2ee7](https://github.com/BenTenmann/protein-classification-service/commit/66e2ee78ea6bd2c1719a6aac8c589ed38b8fcf7d))
* add service deps ([08ddd31](https://github.com/BenTenmann/protein-classification-service/commit/08ddd31cc85777df331dbb83d8e3cb21ad451e01))
* add tests and ci ([08f8732](https://github.com/BenTenmann/protein-classification-service/commit/08f8732365711aed4f39e395d29dabef46b1140b))
* big commit ([cababc3](https://github.com/BenTenmann/protein-classification-service/commit/cababc346b9b558dd0386d89547fa7d51b8681bc))
* big commit 2 ([324e11a](https://github.com/BenTenmann/protein-classification-service/commit/324e11ad529d882484e9fbece3eb179c8163d35a))
* exec doc ([2cea030](https://github.com/BenTenmann/protein-classification-service/commit/2cea0304bd3e3e4c41bd9ac4b2fafbeb09fe21a1))
* model doc ([2f4cb2a](https://github.com/BenTenmann/protein-classification-service/commit/2f4cb2a90df1e5b84728f22fa677511fac47ba53))
* update report ([e8adc8e](https://github.com/BenTenmann/protein-classification-service/commit/e8adc8ed12926ec8e56aa0df4c194b987a45663e))
* utils doc ([ecf4d52](https://github.com/BenTenmann/protein-classification-service/commit/ecf4d523dac8478be9e84129f3110d41d7ae54b2))
