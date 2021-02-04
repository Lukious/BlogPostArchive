

---
title: 'Gatsby로 github 블로그 만들고 배포하기'
date: 2021-02-04 12:00:00
category: 'development'
draft: false
---

# Github blog 바로 시작하기 👋
## 0. 포스트 소개
첫번째 포스트로 Github Blog를 세팅하는 포스트로 하기로 한것은 우선적으로 제가 까먹지 않게 하기 위함이고, 두번째로 딱딱 블로그 세팅만을 다룬 내용의 포스트가 없었기 때문에 작성하기로 하였습니다. 
그렇기 때문에 개념적인 내용은 넘어가고 Practical한 내용 위주로만 포스트가 진행될것 임으로 좀 더 디테일한 개념을 알고 싶으신 분들은 다른 훌륭한 포스트들을 Reference 해볼 수 있겠습니다!

## 1.  Node js 및 npm 설치
우선 제 환경은 Ubuntu 16.04 환경이며 GUI가 없는 text interface 기반 환경입니다.
아래 명령어를 통해 Node.js와 npm설치가 가능합니다. 이후 Gatsby설치까지 한번에 진행 하도록 하겠습니다.
 ```sh
$ sudo apt update
$ sudo apt -y upgrade

$ sudo apt update
$ sudo apt -y install curl 
$ curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -

$ sudo apt -y install nodejs
$ sudo apt -y  install gcc g++ make

$ node --version
v12.18.3
$ npm --ver
6.14.6

$ npm install -g gatsby-cli

```
node --version 명령어와 --ver명령어가 정상적으로 출력 된다면 맞게 설치된 것입니다.

이후 npm intsall -g gatsby-cli를 통해 Gatsby를 설치해 주도록 합시다.

## 2. Gatsby 테마 고르기
[Gatsby starter theme](https://www.gatsbyjs.org/starters/?v=2) 에서 사용하고 싶은 테마를 가져올 수 있습니다.

저같은 경우 현재 이 블로그에 적용된 [# gatsby-starter-bee](https://www.gatsbyjs.com/starters/JaeYeopHan/gatsby-starter-bee/)를 적용하였습니다. 해당 테마를 이용한 이유는 무엇보다 기본적인 기능에 충실하고, 코드 자체가 깔끔해 차후 수정하여 사용하기가 간편했기 때문입니다.
 
 테마를 정하고 스크롤 다운하여 Install this starter locally: 에 있는 명령어를 통해 테마를 다운로드 받습니다 (git clone과 동일합니다).
```sh
$ gatsby new gatsby new gatsby-starter-bee https://github.com/JaeYeopHan/gatsby-starter-bee
$ cd gatsby-starter-bee
$ npm start
```
npm start를 입력하면 build가 진행되며 이후 [http://localhost:8000/](http://localhost:8000/)에서 생성된 페이지를 확인 할 수 있다.
만약 8000번 포트가 이미 사용중이라면 자동으로 사용가능한 포트에서 (예.8001) 오픈되니 각 포트에 맞춰서 접속해 보면 실행되는 것을 확인 할 수 있다.

### gatsby-starter-bee테마에 대해서...
gatsby-starter-bee 테마를 사용한다면 npm start 전에 **gatsby-meta-config.js** 파일을 수정하여 개인화 하는것이 더욱 좋다.
주석에 적혀있는대로 title이나 페이지 주소, google-anlytics Tracking ID 등을 맞게 입력하면 블로그에 필요한 여러 기능에 별다른 작업을 하지 않아도 곧바로 플러그인을 통해 적용된다. 

## 3. Github pages에 배포하기

### 배포를 위한 환경 만들기
gatsby-starter-bee의 경우 기본적으로 [netlify](https://netlify.com/)를 통한 배포를 지원하지만 이번 포스트에서는 일반적인 배포 방법을 바탕으로, 또한 github blog는 username.github.io에 배포하는것이 대부분일테니 이를 적용하여 배포하겠다. 

우선적으로 gh-pages를 설치해 보자.
```sh
$ npm install gh-pages --save-dev
```
deploy(배포 시 사용하는 명령어) 설정을 위해 아래 코드를 package.js에 추가해 준다.
```sh
//package.json
"scripts": {
	"deploy" :"gatsby build && gh-pages -d public -b master",
	"post": "gatsby-post-gen",
	"dev": "gatsby develop",
	...
```
Github에서 배포에 사용할 repository를 새로 만들어준다. 
repository이름은 위에서 설명했듯이 username.github.io (필자같은 경우 lukious.github.io)로 하여 생성하도록 한다. 이때 대소문자는 구분할 필요 없으며 반드시 public으로 생성해야한다.

### 로컬과 연결하여 배포하기
로컬에서 만들어진 페이지와 repository를 연결하여 배포하기 위해 아래의 명령어를 작성해준다.
```sh
$ git remote add origin https://github.com/lukious/lukious.github.io.git
$ git add .
$ git checkout -b develop
$ git commit -m "initial commit"
$ npm run deploy
```
배포 이후 레포지토리 이름인 username.github.io에 접속하면 배포된 페이지를 볼 수 있다.

### 포스트 작성 이후 다음 배포하기
포스트 작성은 각 테마마다 적용 방식이 다름으로 이는 해당 포스트와 완벽히 같지 않을 수 있다. gatsby-starter-bee같은 경우 README.md에 작성된 바와 같이 **content/blog** 에 포스트를 작성하여 업로드 할 수 있다.
마크다운으로 작성한 포스트를 업로드 한뒤 아래 명령어를 통해 새로작성한 페이지와 함께 배포 할 수 있다.
```sh
$ npm start
$ npm run deploy 
```
