FROM node:14.9.0-alpine AS build-env

WORKDIR /app

COPY package.json yarn.lock ./

RUN apk add --no-cache g++ gcc libgcc libstdc++ linux-headers make python

# Install packages
RUN npm install --loglevel=error
# RUN npm install --force --loglevel=error
# RUN yarn add node-sass
# RUN yarn install

COPY . .

RUN npm run build

FROM nginx:stable-alpine

COPY nginx.conf /temp/nginx.conf

RUN envsubst /app < /temp/nginx.conf > /etc/nginx/conf.d/default.conf

COPY --from=build-env /app/dist /usr/share/nginx/html

EXPOSE 80