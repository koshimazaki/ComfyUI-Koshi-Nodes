// @__NO_SIDE_EFFECTS__
function nc(n) {
  const e = /* @__PURE__ */ Object.create(null);
  for (const t of n.split(",")) e[t] = 1;
  return (t) => t in e;
}
const ut = {}, As = [], Fn = () => {
}, Fh = () => !1, Go = (n) => n.charCodeAt(0) === 111 && n.charCodeAt(1) === 110 && // uppercase letter
(n.charCodeAt(2) > 122 || n.charCodeAt(2) < 97), Wo = (n) => n.startsWith("onUpdate:"), Rt = Object.assign, ic = (n, e) => {
  const t = n.indexOf(e);
  t > -1 && n.splice(t, 1);
}, Yd = Object.prototype.hasOwnProperty, it = (n, e) => Yd.call(n, e), ze = Array.isArray, ws = (n) => Cr(n) === "[object Map]", Oh = (n) => Cr(n) === "[object Set]", Wc = (n) => Cr(n) === "[object Date]", Xe = (n) => typeof n == "function", xt = (n) => typeof n == "string", On = (n) => typeof n == "symbol", st = (n) => n !== null && typeof n == "object", Bh = (n) => (st(n) || Xe(n)) && Xe(n.then) && Xe(n.catch), zh = Object.prototype.toString, Cr = (n) => zh.call(n), qd = (n) => Cr(n).slice(8, -1), Hh = (n) => Cr(n) === "[object Object]", sc = (n) => xt(n) && n !== "NaN" && n[0] !== "-" && "" + parseInt(n, 10) === n, ar = /* @__PURE__ */ nc(
  // the leading comma is intentional so empty string "" is also included
  ",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"
), Xo = (n) => {
  const e = /* @__PURE__ */ Object.create(null);
  return ((t) => e[t] || (e[t] = n(t)));
}, jd = /-\w/g, Mn = Xo(
  (n) => n.replace(jd, (e) => e.slice(1).toUpperCase())
), Kd = /\B([A-Z])/g, Ki = Xo(
  (n) => n.replace(Kd, "-$1").toLowerCase()
), Vh = Xo((n) => n.charAt(0).toUpperCase() + n.slice(1)), oa = Xo(
  (n) => n ? `on${Vh(n)}` : ""
), In = (n, e) => !Object.is(n, e), aa = (n, ...e) => {
  for (let t = 0; t < n.length; t++)
    n[t](...e);
}, kh = (n, e, t, i = !1) => {
  Object.defineProperty(n, e, {
    configurable: !0,
    enumerable: !1,
    writable: i,
    value: t
  });
}, $d = (n) => {
  const e = parseFloat(n);
  return isNaN(e) ? n : e;
}, Zd = (n) => {
  const e = xt(n) ? Number(n) : NaN;
  return isNaN(e) ? n : e;
};
let Xc;
const Yo = () => Xc || (Xc = typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {});
function _i(n) {
  if (ze(n)) {
    const e = {};
    for (let t = 0; t < n.length; t++) {
      const i = n[t], s = xt(i) ? tp(i) : _i(i);
      if (s)
        for (const r in s)
          e[r] = s[r];
    }
    return e;
  } else if (xt(n) || st(n))
    return n;
}
const Jd = /;(?![^(]*\))/g, Qd = /:([^]+)/, ep = /\/\*[^]*?\*\//g;
function tp(n) {
  const e = {};
  return n.replace(ep, "").split(Jd).forEach((t) => {
    if (t) {
      const i = t.split(Qd);
      i.length > 1 && (e[i[0].trim()] = i[1].trim());
    }
  }), e;
}
function pr(n) {
  let e = "";
  if (xt(n))
    e = n;
  else if (ze(n))
    for (let t = 0; t < n.length; t++) {
      const i = pr(n[t]);
      i && (e += i + " ");
    }
  else if (st(n))
    for (const t in n)
      n[t] && (e += t + " ");
  return e.trim();
}
const np = "itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly", ip = /* @__PURE__ */ nc(np);
function Gh(n) {
  return !!n || n === "";
}
function sp(n, e) {
  if (n.length !== e.length) return !1;
  let t = !0;
  for (let i = 0; t && i < n.length; i++)
    t = rc(n[i], e[i]);
  return t;
}
function rc(n, e) {
  if (n === e) return !0;
  let t = Wc(n), i = Wc(e);
  if (t || i)
    return t && i ? n.getTime() === e.getTime() : !1;
  if (t = On(n), i = On(e), t || i)
    return n === e;
  if (t = ze(n), i = ze(e), t || i)
    return t && i ? sp(n, e) : !1;
  if (t = st(n), i = st(e), t || i) {
    if (!t || !i)
      return !1;
    const s = Object.keys(n).length, r = Object.keys(e).length;
    if (s !== r)
      return !1;
    for (const o in n) {
      const a = n.hasOwnProperty(o), l = e.hasOwnProperty(o);
      if (a && !l || !a && l || !rc(n[o], e[o]))
        return !1;
    }
  }
  return String(n) === String(e);
}
const Wh = (n) => !!(n && n.__v_isRef === !0), cn = (n) => xt(n) ? n : n == null ? "" : ze(n) || st(n) && (n.toString === zh || !Xe(n.toString)) ? Wh(n) ? cn(n.value) : JSON.stringify(n, Xh, 2) : String(n), Xh = (n, e) => Wh(e) ? Xh(n, e.value) : ws(e) ? {
  [`Map(${e.size})`]: [...e.entries()].reduce(
    (t, [i, s], r) => (t[la(i, r) + " =>"] = s, t),
    {}
  )
} : Oh(e) ? {
  [`Set(${e.size})`]: [...e.values()].map((t) => la(t))
} : On(e) ? la(e) : st(e) && !ze(e) && !Hh(e) ? String(e) : e, la = (n, e = "") => {
  var t;
  return (
    // Symbol.description in es2019+ so we need to cast here to pass
    // the lib: es2016 check
    On(n) ? `Symbol(${(t = n.description) != null ? t : e})` : n
  );
};
let Ct;
class rp {
  // TODO isolatedDeclarations "__v_skip"
  constructor(e = !1) {
    this.detached = e, this._active = !0, this._on = 0, this.effects = [], this.cleanups = [], this._isPaused = !1, this._warnOnRun = !0, this.__v_skip = !0, !e && Ct && (Ct.active ? (this.parent = Ct, this.index = (Ct.scopes || (Ct.scopes = [])).push(
      this
    ) - 1) : (this._active = !1, this._warnOnRun = !1));
  }
  get active() {
    return this._active;
  }
  pause() {
    if (this._active) {
      this._isPaused = !0;
      let e, t;
      if (this.scopes) {
        const i = this.scopes.slice();
        for (e = 0, t = i.length; e < t; e++)
          i[e].pause();
      }
      for (e = 0, t = this.effects.length; e < t; e++)
        this.effects[e].pause();
    }
  }
  /**
   * Resumes the effect scope, including all child scopes and effects.
   */
  resume() {
    if (this._active && this._isPaused) {
      this._isPaused = !1;
      let e, t;
      if (this.scopes) {
        const s = this.scopes.slice();
        for (e = 0, t = s.length; e < t; e++)
          s[e].resume();
      }
      const i = this.effects.slice();
      for (e = 0, t = i.length; e < t; e++)
        i[e].resume();
    }
  }
  run(e) {
    if (this._active) {
      const t = Ct;
      try {
        return Ct = this, e();
      } finally {
        Ct = t;
      }
    }
  }
  /**
   * This should only be called on non-detached scopes
   * @internal
   */
  on() {
    ++this._on === 1 && (this.prevScope = Ct, Ct = this);
  }
  /**
   * This should only be called on non-detached scopes
   * @internal
   */
  off() {
    if (this._on > 0 && --this._on === 0) {
      if (Ct === this)
        Ct = this.prevScope;
      else {
        let e = Ct;
        for (; e; ) {
          if (e.prevScope === this) {
            e.prevScope = this.prevScope;
            break;
          }
          e = e.prevScope;
        }
      }
      this.prevScope = void 0;
    }
  }
  stop(e) {
    if (this._active) {
      this._active = !1;
      let t, i;
      for (t = 0, i = this.effects.length; t < i; t++)
        this.effects[t].stop();
      for (this.effects.length = 0, t = 0, i = this.cleanups.length; t < i; t++)
        this.cleanups[t]();
      if (this.cleanups.length = 0, this.scopes) {
        const s = this.scopes.slice();
        for (t = 0, i = s.length; t < i; t++)
          s[t].stop(!0);
        this.scopes.length = 0;
      }
      if (!this.detached && this.parent && !e) {
        const s = this.parent.scopes.pop();
        s && s !== this && (this.parent.scopes[this.index] = s, s.index = this.index);
      }
      this.parent = void 0;
    }
  }
}
function op() {
  return Ct;
}
let ft;
const ca = /* @__PURE__ */ new WeakSet();
class Yh {
  constructor(e) {
    this.fn = e, this.deps = void 0, this.depsTail = void 0, this.flags = 5, this.next = void 0, this.cleanup = void 0, this.scheduler = void 0, Ct && (Ct.active ? Ct.effects.push(this) : this.flags &= -2);
  }
  pause() {
    this.flags |= 64;
  }
  resume() {
    this.flags & 64 && (this.flags &= -65, ca.has(this) && (ca.delete(this), this.trigger()));
  }
  /**
   * @internal
   */
  notify() {
    this.flags & 2 && !(this.flags & 32) || this.flags & 8 || jh(this);
  }
  run() {
    if (!(this.flags & 1))
      return this.fn();
    this.flags |= 2, Yc(this), Kh(this);
    const e = ft, t = Sn;
    ft = this, Sn = !0;
    try {
      return this.fn();
    } finally {
      $h(this), ft = e, Sn = t, this.flags &= -3;
    }
  }
  stop() {
    if (this.flags & 1) {
      for (let e = this.deps; e; e = e.nextDep)
        lc(e);
      this.deps = this.depsTail = void 0, Yc(this), this.onStop && this.onStop(), this.flags &= -2;
    }
  }
  trigger() {
    this.flags & 64 ? ca.add(this) : this.scheduler ? this.scheduler() : this.runIfDirty();
  }
  /**
   * @internal
   */
  runIfDirty() {
    tl(this) && this.run();
  }
  get dirty() {
    return tl(this);
  }
}
let qh = 0, lr, cr;
function jh(n, e = !1) {
  if (n.flags |= 8, e) {
    n.next = cr, cr = n;
    return;
  }
  n.next = lr, lr = n;
}
function oc() {
  qh++;
}
function ac() {
  if (--qh > 0)
    return;
  if (cr) {
    let e = cr;
    for (cr = void 0; e; ) {
      const t = e.next;
      e.next = void 0, e.flags &= -9, e = t;
    }
  }
  let n;
  for (; lr; ) {
    let e = lr;
    for (lr = void 0; e; ) {
      const t = e.next;
      if (e.next = void 0, e.flags &= -9, e.flags & 1)
        try {
          e.trigger();
        } catch (i) {
          n || (n = i);
        }
      e = t;
    }
  }
  if (n) throw n;
}
function Kh(n) {
  for (let e = n.deps; e; e = e.nextDep)
    e.version = -1, e.prevActiveLink = e.dep.activeLink, e.dep.activeLink = e;
}
function $h(n) {
  let e, t = n.depsTail, i = t;
  for (; i; ) {
    const s = i.prevDep;
    i.version === -1 ? (i === t && (t = s), lc(i), ap(i)) : e = i, i.dep.activeLink = i.prevActiveLink, i.prevActiveLink = void 0, i = s;
  }
  n.deps = e, n.depsTail = t;
}
function tl(n) {
  for (let e = n.deps; e; e = e.nextDep)
    if (e.dep.version !== e.version || e.dep.computed && (Zh(e.dep.computed) || e.dep.version !== e.version))
      return !0;
  return !!n._dirty;
}
function Zh(n) {
  if (n.flags & 4 && !(n.flags & 16) || (n.flags &= -17, n.globalVersion === mr) || (n.globalVersion = mr, !n.isSSR && n.flags & 128 && (!n.deps && !n._dirty || !tl(n))))
    return;
  n.flags |= 2;
  const e = n.dep, t = ft, i = Sn;
  ft = n, Sn = !0;
  try {
    Kh(n);
    const s = n.fn(n._value);
    (e.version === 0 || In(s, n._value)) && (n.flags |= 128, n._value = s, e.version++);
  } catch (s) {
    throw e.version++, s;
  } finally {
    ft = t, Sn = i, $h(n), n.flags &= -3;
  }
}
function lc(n, e = !1) {
  const { dep: t, prevSub: i, nextSub: s } = n;
  if (i && (i.nextSub = s, n.prevSub = void 0), s && (s.prevSub = i, n.nextSub = void 0), t.subs === n && (t.subs = i, !i && t.computed)) {
    t.computed.flags &= -5;
    for (let r = t.computed.deps; r; r = r.nextDep)
      lc(r, !0);
  }
  !e && !--t.sc && t.map && t.map.delete(t.key);
}
function ap(n) {
  const { prevDep: e, nextDep: t } = n;
  e && (e.nextDep = t, n.prevDep = void 0), t && (t.prevDep = e, n.nextDep = void 0);
}
let Sn = !0;
const Jh = [];
function ni() {
  Jh.push(Sn), Sn = !1;
}
function ii() {
  const n = Jh.pop();
  Sn = n === void 0 ? !0 : n;
}
function Yc(n) {
  const { cleanup: e } = n;
  if (n.cleanup = void 0, e) {
    const t = ft;
    ft = void 0;
    try {
      e();
    } finally {
      ft = t;
    }
  }
}
let mr = 0;
class lp {
  constructor(e, t) {
    this.sub = e, this.dep = t, this.version = t.version, this.nextDep = this.prevDep = this.nextSub = this.prevSub = this.prevActiveLink = void 0;
  }
}
class cc {
  // TODO isolatedDeclarations "__v_skip"
  constructor(e) {
    this.computed = e, this.version = 0, this.activeLink = void 0, this.subs = void 0, this.map = void 0, this.key = void 0, this.sc = 0, this.__v_skip = !0;
  }
  track(e) {
    if (!ft || !Sn || ft === this.computed)
      return;
    let t = this.activeLink;
    if (t === void 0 || t.sub !== ft)
      t = this.activeLink = new lp(ft, this), ft.deps ? (t.prevDep = ft.depsTail, ft.depsTail.nextDep = t, ft.depsTail = t) : ft.deps = ft.depsTail = t, Qh(t);
    else if (t.version === -1 && (t.version = this.version, t.nextDep)) {
      const i = t.nextDep;
      i.prevDep = t.prevDep, t.prevDep && (t.prevDep.nextDep = i), t.prevDep = ft.depsTail, t.nextDep = void 0, ft.depsTail.nextDep = t, ft.depsTail = t, ft.deps === t && (ft.deps = i);
    }
    return t;
  }
  trigger(e) {
    this.version++, mr++, this.notify(e);
  }
  notify(e) {
    oc();
    try {
      for (let t = this.subs; t; t = t.prevSub)
        t.sub.notify() && t.sub.dep.notify();
    } finally {
      ac();
    }
  }
}
function Qh(n) {
  if (n.dep.sc++, n.sub.flags & 4) {
    const e = n.dep.computed;
    if (e && !n.dep.subs) {
      e.flags |= 20;
      for (let i = e.deps; i; i = i.nextDep)
        Qh(i);
    }
    const t = n.dep.subs;
    t !== n && (n.prevSub = t, t && (t.nextSub = n)), n.dep.subs = n;
  }
}
const nl = /* @__PURE__ */ new WeakMap(), ki = /* @__PURE__ */ Symbol(
  ""
), il = /* @__PURE__ */ Symbol(
  ""
), _r = /* @__PURE__ */ Symbol(
  ""
);
function It(n, e, t) {
  if (Sn && ft) {
    let i = nl.get(n);
    i || nl.set(n, i = /* @__PURE__ */ new Map());
    let s = i.get(t);
    s || (i.set(t, s = new cc()), s.map = i, s.key = t), s.track();
  }
}
function Zn(n, e, t, i, s, r) {
  const o = nl.get(n);
  if (!o) {
    mr++;
    return;
  }
  const a = (l) => {
    l && l.trigger();
  };
  if (oc(), e === "clear")
    o.forEach(a);
  else {
    const l = ze(n), c = l && sc(t);
    if (l && t === "length") {
      const u = Number(i);
      o.forEach((h, f) => {
        (f === "length" || f === _r || !On(f) && f >= u) && a(h);
      });
    } else
      switch ((t !== void 0 || o.has(void 0)) && a(o.get(t)), c && a(o.get(_r)), e) {
        case "add":
          l ? c && a(o.get("length")) : (a(o.get(ki)), ws(n) && a(o.get(il)));
          break;
        case "delete":
          l || (a(o.get(ki)), ws(n) && a(o.get(il)));
          break;
        case "set":
          ws(n) && a(o.get(ki));
          break;
      }
  }
  ac();
}
function es(n) {
  const e = /* @__PURE__ */ et(n);
  return e === n ? e : (It(e, "iterate", _r), /* @__PURE__ */ pn(n) ? e : e.map(Tn));
}
function qo(n) {
  return It(n = /* @__PURE__ */ et(n), "iterate", _r), n;
}
function Cn(n, e) {
  return /* @__PURE__ */ si(n) ? Is(/* @__PURE__ */ Gi(n) ? Tn(e) : e) : Tn(e);
}
const cp = {
  __proto__: null,
  [Symbol.iterator]() {
    return ua(this, Symbol.iterator, (n) => Cn(this, n));
  },
  concat(...n) {
    return es(this).concat(
      ...n.map((e) => ze(e) ? es(e) : e)
    );
  },
  entries() {
    return ua(this, "entries", (n) => (n[1] = Cn(this, n[1]), n));
  },
  every(n, e) {
    return Vn(this, "every", n, e, void 0, arguments);
  },
  filter(n, e) {
    return Vn(
      this,
      "filter",
      n,
      e,
      (t) => t.map((i) => Cn(this, i)),
      arguments
    );
  },
  find(n, e) {
    return Vn(
      this,
      "find",
      n,
      e,
      (t) => Cn(this, t),
      arguments
    );
  },
  findIndex(n, e) {
    return Vn(this, "findIndex", n, e, void 0, arguments);
  },
  findLast(n, e) {
    return Vn(
      this,
      "findLast",
      n,
      e,
      (t) => Cn(this, t),
      arguments
    );
  },
  findLastIndex(n, e) {
    return Vn(this, "findLastIndex", n, e, void 0, arguments);
  },
  // flat, flatMap could benefit from ARRAY_ITERATE but are not straight-forward to implement
  forEach(n, e) {
    return Vn(this, "forEach", n, e, void 0, arguments);
  },
  includes(...n) {
    return ha(this, "includes", n);
  },
  indexOf(...n) {
    return ha(this, "indexOf", n);
  },
  join(n) {
    return es(this).join(n);
  },
  // keys() iterator only reads `length`, no optimization required
  lastIndexOf(...n) {
    return ha(this, "lastIndexOf", n);
  },
  map(n, e) {
    return Vn(this, "map", n, e, void 0, arguments);
  },
  pop() {
    return Gs(this, "pop");
  },
  push(...n) {
    return Gs(this, "push", n);
  },
  reduce(n, ...e) {
    return qc(this, "reduce", n, e);
  },
  reduceRight(n, ...e) {
    return qc(this, "reduceRight", n, e);
  },
  shift() {
    return Gs(this, "shift");
  },
  // slice could use ARRAY_ITERATE but also seems to beg for range tracking
  some(n, e) {
    return Vn(this, "some", n, e, void 0, arguments);
  },
  splice(...n) {
    return Gs(this, "splice", n);
  },
  toReversed() {
    return es(this).toReversed();
  },
  toSorted(n) {
    return es(this).toSorted(n);
  },
  toSpliced(...n) {
    return es(this).toSpliced(...n);
  },
  unshift(...n) {
    return Gs(this, "unshift", n);
  },
  values() {
    return ua(this, "values", (n) => Cn(this, n));
  }
};
function ua(n, e, t) {
  const i = qo(n), s = i[e]();
  return i !== n && !/* @__PURE__ */ pn(n) && (s._next = s.next, s.next = () => {
    const r = s._next();
    return r.done || (r.value = t(r.value)), r;
  }), s;
}
const up = Array.prototype;
function Vn(n, e, t, i, s, r) {
  const o = qo(n), a = o !== n && !/* @__PURE__ */ pn(n), l = o[e];
  if (l !== up[e]) {
    const h = l.apply(n, r);
    return a ? Tn(h) : h;
  }
  let c = t;
  o !== n && (a ? c = function(h, f) {
    return t.call(this, Cn(n, h), f, n);
  } : t.length > 2 && (c = function(h, f) {
    return t.call(this, h, f, n);
  }));
  const u = l.call(o, c, i);
  return a && s ? s(u) : u;
}
function qc(n, e, t, i) {
  const s = qo(n), r = s !== n && !/* @__PURE__ */ pn(n);
  let o = t, a = !1;
  s !== n && (r ? (a = i.length === 0, o = function(c, u, h) {
    return a && (a = !1, c = Cn(n, c)), t.call(this, c, Cn(n, u), h, n);
  }) : t.length > 3 && (o = function(c, u, h) {
    return t.call(this, c, u, h, n);
  }));
  const l = s[e](o, ...i);
  return a ? Cn(n, l) : l;
}
function ha(n, e, t) {
  const i = /* @__PURE__ */ et(n);
  It(i, "iterate", _r);
  const s = i[e](...t);
  return (s === -1 || s === !1) && /* @__PURE__ */ fc(t[0]) ? (t[0] = /* @__PURE__ */ et(t[0]), i[e](...t)) : s;
}
function Gs(n, e, t = []) {
  ni(), oc();
  const i = (/* @__PURE__ */ et(n))[e].apply(n, t);
  return ac(), ii(), i;
}
const hp = /* @__PURE__ */ nc("__proto__,__v_isRef,__isVue"), ef = new Set(
  /* @__PURE__ */ Object.getOwnPropertyNames(Symbol).filter((n) => n !== "arguments" && n !== "caller").map((n) => Symbol[n]).filter(On)
);
function fp(n) {
  On(n) || (n = String(n));
  const e = /* @__PURE__ */ et(this);
  return It(e, "has", n), e.hasOwnProperty(n);
}
class tf {
  constructor(e = !1, t = !1) {
    this._isReadonly = e, this._isShallow = t;
  }
  get(e, t, i) {
    if (t === "__v_skip") return e.__v_skip;
    const s = this._isReadonly, r = this._isShallow;
    if (t === "__v_isReactive")
      return !s;
    if (t === "__v_isReadonly")
      return s;
    if (t === "__v_isShallow")
      return r;
    if (t === "__v_raw")
      return i === (s ? r ? yp : of : r ? rf : sf).get(e) || // receiver is not the reactive proxy, but has the same prototype
      // this means the receiver is a user proxy of the reactive proxy
      Object.getPrototypeOf(e) === Object.getPrototypeOf(i) ? e : void 0;
    const o = ze(e);
    if (!s) {
      let l;
      if (o && (l = cp[t]))
        return l;
      if (t === "hasOwnProperty")
        return fp;
    }
    const a = Reflect.get(
      e,
      t,
      // if this is a proxy wrapping a ref, return methods using the raw ref
      // as receiver so that we don't have to call `toRaw` on the ref in all
      // its class methods
      /* @__PURE__ */ Ut(e) ? e : i
    );
    if ((On(t) ? ef.has(t) : hp(t)) || (s || It(e, "get", t), r))
      return a;
    if (/* @__PURE__ */ Ut(a)) {
      const l = o && sc(t) ? a : a.value;
      return s && st(l) ? /* @__PURE__ */ rl(l) : l;
    }
    return st(a) ? s ? /* @__PURE__ */ rl(a) : /* @__PURE__ */ gr(a) : a;
  }
}
class nf extends tf {
  constructor(e = !1) {
    super(!1, e);
  }
  set(e, t, i, s) {
    let r = e[t];
    const o = ze(e) && sc(t);
    if (!this._isShallow) {
      const c = /* @__PURE__ */ si(r);
      if (!/* @__PURE__ */ pn(i) && !/* @__PURE__ */ si(i) && (r = /* @__PURE__ */ et(r), i = /* @__PURE__ */ et(i)), !o && /* @__PURE__ */ Ut(r) && !/* @__PURE__ */ Ut(i))
        return c || (r.value = i), !0;
    }
    const a = o ? Number(t) < e.length : it(e, t), l = Reflect.set(
      e,
      t,
      i,
      /* @__PURE__ */ Ut(e) ? e : s
    );
    return e === /* @__PURE__ */ et(s) && l && (a ? In(i, r) && Zn(e, "set", t, i) : Zn(e, "add", t, i)), l;
  }
  deleteProperty(e, t) {
    const i = it(e, t);
    e[t];
    const s = Reflect.deleteProperty(e, t);
    return s && i && Zn(e, "delete", t, void 0), s;
  }
  has(e, t) {
    const i = Reflect.has(e, t);
    return (!On(t) || !ef.has(t)) && It(e, "has", t), i;
  }
  ownKeys(e) {
    return It(
      e,
      "iterate",
      ze(e) ? "length" : ki
    ), Reflect.ownKeys(e);
  }
}
class dp extends tf {
  constructor(e = !1) {
    super(!0, e);
  }
  set(e, t) {
    return !0;
  }
  deleteProperty(e, t) {
    return !0;
  }
}
const pp = /* @__PURE__ */ new nf(), mp = /* @__PURE__ */ new dp(), _p = /* @__PURE__ */ new nf(!0);
const sl = (n) => n, Br = (n) => Reflect.getPrototypeOf(n);
function gp(n, e, t) {
  return function(...i) {
    const s = this.__v_raw, r = /* @__PURE__ */ et(s), o = ws(r), a = n === "entries" || n === Symbol.iterator && o, l = n === "keys" && o, c = s[n](...i), u = t ? sl : e ? Is : Tn;
    return !e && It(
      r,
      "iterate",
      l ? il : ki
    ), Rt(
      // inheriting all iterator properties
      Object.create(c),
      {
        // iterator protocol
        next() {
          const { value: h, done: f } = c.next();
          return f ? { value: h, done: f } : {
            value: a ? [u(h[0]), u(h[1])] : u(h),
            done: f
          };
        }
      }
    );
  };
}
function zr(n) {
  return function(...e) {
    return n === "delete" ? !1 : n === "clear" ? void 0 : this;
  };
}
function vp(n, e) {
  const t = {
    get(s) {
      const r = this.__v_raw, o = /* @__PURE__ */ et(r), a = /* @__PURE__ */ et(s);
      n || (In(s, a) && It(o, "get", s), It(o, "get", a));
      const { has: l } = Br(o), c = e ? sl : n ? Is : Tn;
      if (l.call(o, s))
        return c(r.get(s));
      if (l.call(o, a))
        return c(r.get(a));
      r !== o && r.get(s);
    },
    get size() {
      const s = this.__v_raw;
      return !n && It(/* @__PURE__ */ et(s), "iterate", ki), s.size;
    },
    has(s) {
      const r = this.__v_raw, o = /* @__PURE__ */ et(r), a = /* @__PURE__ */ et(s);
      return n || (In(s, a) && It(o, "has", s), It(o, "has", a)), s === a ? r.has(s) : r.has(s) || r.has(a);
    },
    forEach(s, r) {
      const o = this, a = o.__v_raw, l = /* @__PURE__ */ et(a), c = e ? sl : n ? Is : Tn;
      return !n && It(l, "iterate", ki), a.forEach((u, h) => s.call(r, c(u), c(h), o));
    }
  };
  return Rt(
    t,
    n ? {
      add: zr("add"),
      set: zr("set"),
      delete: zr("delete"),
      clear: zr("clear")
    } : {
      add(s) {
        const r = /* @__PURE__ */ et(this), o = Br(r), a = /* @__PURE__ */ et(s), l = !e && !/* @__PURE__ */ pn(s) && !/* @__PURE__ */ si(s) ? a : s;
        return o.has.call(r, l) || In(s, l) && o.has.call(r, s) || In(a, l) && o.has.call(r, a) || (r.add(l), Zn(r, "add", l, l)), this;
      },
      set(s, r) {
        !e && !/* @__PURE__ */ pn(r) && !/* @__PURE__ */ si(r) && (r = /* @__PURE__ */ et(r));
        const o = /* @__PURE__ */ et(this), { has: a, get: l } = Br(o);
        let c = a.call(o, s);
        c || (s = /* @__PURE__ */ et(s), c = a.call(o, s));
        const u = l.call(o, s);
        return o.set(s, r), c ? In(r, u) && Zn(o, "set", s, r) : Zn(o, "add", s, r), this;
      },
      delete(s) {
        const r = /* @__PURE__ */ et(this), { has: o, get: a } = Br(r);
        let l = o.call(r, s);
        l || (s = /* @__PURE__ */ et(s), l = o.call(r, s)), a && a.call(r, s);
        const c = r.delete(s);
        return l && Zn(r, "delete", s, void 0), c;
      },
      clear() {
        const s = /* @__PURE__ */ et(this), r = s.size !== 0, o = s.clear();
        return r && Zn(
          s,
          "clear",
          void 0,
          void 0
        ), o;
      }
    }
  ), [
    "keys",
    "values",
    "entries",
    Symbol.iterator
  ].forEach((s) => {
    t[s] = gp(s, n, e);
  }), t;
}
function uc(n, e) {
  const t = vp(n, e);
  return (i, s, r) => s === "__v_isReactive" ? !n : s === "__v_isReadonly" ? n : s === "__v_raw" ? i : Reflect.get(
    it(t, s) && s in i ? t : i,
    s,
    r
  );
}
const xp = {
  get: /* @__PURE__ */ uc(!1, !1)
}, Mp = {
  get: /* @__PURE__ */ uc(!1, !0)
}, Sp = {
  get: /* @__PURE__ */ uc(!0, !1)
};
const sf = /* @__PURE__ */ new WeakMap(), rf = /* @__PURE__ */ new WeakMap(), of = /* @__PURE__ */ new WeakMap(), yp = /* @__PURE__ */ new WeakMap();
function Ep(n) {
  switch (n) {
    case "Object":
    case "Array":
      return 1;
    case "Map":
    case "Set":
    case "WeakMap":
    case "WeakSet":
      return 2;
    default:
      return 0;
  }
}
// @__NO_SIDE_EFFECTS__
function gr(n) {
  return /* @__PURE__ */ si(n) ? n : hc(
    n,
    !1,
    pp,
    xp,
    sf
  );
}
// @__NO_SIDE_EFFECTS__
function Tp(n) {
  return hc(
    n,
    !1,
    _p,
    Mp,
    rf
  );
}
// @__NO_SIDE_EFFECTS__
function rl(n) {
  return hc(
    n,
    !0,
    mp,
    Sp,
    of
  );
}
function hc(n, e, t, i, s) {
  if (!st(n) || n.__v_raw && !(e && n.__v_isReactive) || n.__v_skip || !Object.isExtensible(n))
    return n;
  const r = s.get(n);
  if (r)
    return r;
  const o = Ep(qd(n));
  if (o === 0)
    return n;
  const a = new Proxy(
    n,
    o === 2 ? i : t
  );
  return s.set(n, a), a;
}
// @__NO_SIDE_EFFECTS__
function Gi(n) {
  return /* @__PURE__ */ si(n) ? /* @__PURE__ */ Gi(n.__v_raw) : !!(n && n.__v_isReactive);
}
// @__NO_SIDE_EFFECTS__
function si(n) {
  return !!(n && n.__v_isReadonly);
}
// @__NO_SIDE_EFFECTS__
function pn(n) {
  return !!(n && n.__v_isShallow);
}
// @__NO_SIDE_EFFECTS__
function fc(n) {
  return n ? !!n.__v_raw : !1;
}
// @__NO_SIDE_EFFECTS__
function et(n) {
  const e = n && n.__v_raw;
  return e ? /* @__PURE__ */ et(e) : n;
}
function bp(n) {
  return !it(n, "__v_skip") && Object.isExtensible(n) && kh(n, "__v_skip", !0), n;
}
const Tn = (n) => st(n) ? /* @__PURE__ */ gr(n) : n, Is = (n) => st(n) ? /* @__PURE__ */ rl(n) : n;
// @__NO_SIDE_EFFECTS__
function Ut(n) {
  return n ? n.__v_isRef === !0 : !1;
}
// @__NO_SIDE_EFFECTS__
function Ws(n) {
  return Ap(n, !1);
}
function Ap(n, e) {
  return /* @__PURE__ */ Ut(n) ? n : new wp(n, e);
}
class wp {
  constructor(e, t) {
    this.dep = new cc(), this.__v_isRef = !0, this.__v_isShallow = !1, this._rawValue = t ? e : /* @__PURE__ */ et(e), this._value = t ? e : Tn(e), this.__v_isShallow = t;
  }
  get value() {
    return this.dep.track(), this._value;
  }
  set value(e) {
    const t = this._rawValue, i = this.__v_isShallow || /* @__PURE__ */ pn(e) || /* @__PURE__ */ si(e);
    e = i ? e : /* @__PURE__ */ et(e), In(e, t) && (this._rawValue = e, this._value = i ? e : Tn(e), this.dep.trigger());
  }
}
function yt(n) {
  return /* @__PURE__ */ Ut(n) ? n.value : n;
}
const Rp = {
  get: (n, e, t) => e === "__v_raw" ? n : yt(Reflect.get(n, e, t)),
  set: (n, e, t, i) => {
    const s = n[e];
    return /* @__PURE__ */ Ut(s) && !/* @__PURE__ */ Ut(t) ? (s.value = t, !0) : Reflect.set(n, e, t, i);
  }
};
function af(n) {
  return /* @__PURE__ */ Gi(n) ? n : new Proxy(n, Rp);
}
class Cp {
  constructor(e, t, i) {
    this.fn = e, this.setter = t, this._value = void 0, this.dep = new cc(this), this.__v_isRef = !0, this.deps = void 0, this.depsTail = void 0, this.flags = 16, this.globalVersion = mr - 1, this.next = void 0, this.effect = this, this.__v_isReadonly = !t, this.isSSR = i;
  }
  /**
   * @internal
   */
  notify() {
    if (this.flags |= 16, !(this.flags & 8) && // avoid infinite self recursion
    ft !== this)
      return jh(this, !0), !0;
  }
  get value() {
    const e = this.dep.track();
    return Zh(this), e && (e.version = this.dep.version), this._value;
  }
  set value(e) {
    this.setter && this.setter(e);
  }
}
// @__NO_SIDE_EFFECTS__
function Pp(n, e, t = !1) {
  let i, s;
  return Xe(n) ? i = n : (i = n.get, s = n.set), new Cp(i, s, t);
}
const Hr = {}, wo = /* @__PURE__ */ new WeakMap();
let Ni;
function Dp(n, e = !1, t = Ni) {
  if (t) {
    let i = wo.get(t);
    i || wo.set(t, i = []), i.push(n);
  }
}
function Lp(n, e, t = ut) {
  const { immediate: i, deep: s, once: r, scheduler: o, augmentJob: a, call: l } = t, c = (M) => s ? M : /* @__PURE__ */ pn(M) || s === !1 || s === 0 ? Jn(M, 1) : Jn(M);
  let u, h, f, p, v = !1, x = !1;
  if (/* @__PURE__ */ Ut(n) ? (h = () => n.value, v = /* @__PURE__ */ pn(n)) : /* @__PURE__ */ Gi(n) ? (h = () => c(n), v = !0) : ze(n) ? (x = !0, v = n.some((M) => /* @__PURE__ */ Gi(M) || /* @__PURE__ */ pn(M)), h = () => n.map((M) => {
    if (/* @__PURE__ */ Ut(M))
      return M.value;
    if (/* @__PURE__ */ Gi(M))
      return c(M);
    if (Xe(M))
      return l ? l(M, 2) : M();
  })) : Xe(n) ? e ? h = l ? () => l(n, 2) : n : h = () => {
    if (f) {
      ni();
      try {
        f();
      } finally {
        ii();
      }
    }
    const M = Ni;
    Ni = u;
    try {
      return l ? l(n, 3, [p]) : n(p);
    } finally {
      Ni = M;
    }
  } : h = Fn, e && s) {
    const M = h, R = s === !0 ? 1 / 0 : s;
    h = () => Jn(M(), R);
  }
  const m = op(), d = () => {
    u.stop(), m && m.active && ic(m.effects, u);
  };
  if (r && e) {
    const M = e;
    e = (...R) => {
      const w = M(...R);
      return d(), w;
    };
  }
  let b = x ? new Array(n.length).fill(Hr) : Hr;
  const A = (M) => {
    if (!(!(u.flags & 1) || !u.dirty && !M))
      if (e) {
        const R = u.run();
        if (M || s || v || (x ? R.some((w, D) => In(w, b[D])) : In(R, b))) {
          f && f();
          const w = Ni;
          Ni = u;
          try {
            const D = [
              R,
              // pass undefined as the old value when it's changed for the first time
              b === Hr ? void 0 : x && b[0] === Hr ? [] : b,
              p
            ];
            b = R, l ? l(e, 3, D) : (
              // @ts-expect-error
              e(...D)
            );
          } finally {
            Ni = w;
          }
        }
      } else
        u.run();
  };
  return a && a(A), u = new Yh(h), u.scheduler = o ? () => o(A, !1) : A, p = (M) => Dp(M, !1, u), f = u.onStop = () => {
    const M = wo.get(u);
    if (M) {
      if (l)
        l(M, 4);
      else
        for (const R of M) R();
      wo.delete(u);
    }
  }, e ? i ? A(!0) : b = u.run() : o ? o(A.bind(null, !0), !0) : u.run(), d.pause = u.pause.bind(u), d.resume = u.resume.bind(u), d.stop = d, d;
}
function Jn(n, e = 1 / 0, t) {
  if (e <= 0 || !st(n) || n.__v_skip || (t = t || /* @__PURE__ */ new Map(), (t.get(n) || 0) >= e))
    return n;
  if (t.set(n, e), e--, /* @__PURE__ */ Ut(n))
    Jn(n.value, e, t);
  else if (ze(n))
    for (let i = 0; i < n.length; i++)
      Jn(n[i], e, t);
  else if (Oh(n) || ws(n))
    n.forEach((i) => {
      Jn(i, e, t);
    });
  else if (Hh(n)) {
    for (const i in n)
      Jn(n[i], e, t);
    for (const i of Object.getOwnPropertySymbols(n))
      Object.prototype.propertyIsEnumerable.call(n, i) && Jn(n[i], e, t);
  }
  return n;
}
function Pr(n, e, t, i) {
  try {
    return i ? n(...i) : n();
  } catch (s) {
    jo(s, e, t);
  }
}
function mn(n, e, t, i) {
  if (Xe(n)) {
    const s = Pr(n, e, t, i);
    return s && Bh(s) && s.catch((r) => {
      jo(r, e, t);
    }), s;
  }
  if (ze(n)) {
    const s = [];
    for (let r = 0; r < n.length; r++)
      s.push(mn(n[r], e, t, i));
    return s;
  }
}
function jo(n, e, t, i = !0) {
  const s = e ? e.vnode : null, { errorHandler: r, throwUnhandledErrorInProduction: o } = e && e.appContext.config || ut;
  if (e) {
    let a = e.parent;
    const l = e.proxy, c = `https://vuejs.org/error-reference/#runtime-${t}`;
    for (; a; ) {
      const u = a.ec;
      if (u) {
        for (let h = 0; h < u.length; h++)
          if (u[h](n, l, c) === !1)
            return;
      }
      a = a.parent;
    }
    if (r) {
      ni(), Pr(r, null, 10, [
        n,
        l,
        c
      ]), ii();
      return;
    }
  }
  Ip(n, t, s, i, o);
}
function Ip(n, e, t, i = !0, s = !1) {
  if (s)
    throw n;
  console.error(n);
}
const Ht = [];
let wn = -1;
const Rs = [];
let di = null, gs = 0;
const lf = /* @__PURE__ */ Promise.resolve();
let Ro = null;
function Up(n) {
  const e = Ro || lf;
  return n ? e.then(this ? n.bind(this) : n) : e;
}
function Np(n) {
  let e = wn + 1, t = Ht.length;
  for (; e < t; ) {
    const i = e + t >>> 1, s = Ht[i], r = vr(s);
    r < n || r === n && s.flags & 2 ? e = i + 1 : t = i;
  }
  return e;
}
function dc(n) {
  if (!(n.flags & 1)) {
    const e = vr(n), t = Ht[Ht.length - 1];
    !t || // fast path when the job id is larger than the tail
    !(n.flags & 2) && e >= vr(t) ? Ht.push(n) : Ht.splice(Np(e), 0, n), n.flags |= 1, cf();
  }
}
function cf() {
  Ro || (Ro = lf.then(hf));
}
function Fp(n) {
  if (!ze(n))
    di && n.id === -1 ? di.splice(gs + 1, 0, n) : n.flags & 1 || (Rs.push(n), n.flags |= 1);
  else
    for (let e = 0; e < n.length; e++)
      Rs.push(n[e]);
  cf();
}
function jc(n, e, t = wn + 1) {
  for (; t < Ht.length; t++) {
    const i = Ht[t];
    if (i && i.flags & 2) {
      if (n && i.id !== n.uid)
        continue;
      Ht.splice(t, 1), t--, i.flags & 4 && (i.flags &= -2), i(), i.flags & 4 || (i.flags &= -2);
    }
  }
}
function uf(n) {
  if (Rs.length) {
    const e = [...new Set(Rs)].sort(
      (t, i) => vr(t) - vr(i)
    );
    if (Rs.length = 0, di) {
      for (let t = 0; t < e.length; t++)
        di.push(e[t]);
      return;
    }
    for (di = e, gs = 0; gs < di.length; gs++) {
      const t = di[gs];
      t.flags & 4 && (t.flags &= -2), t.flags & 8 || t(), t.flags &= -2;
    }
    di = null, gs = 0;
  }
}
const vr = (n) => n.id == null ? n.flags & 2 ? -1 : 1 / 0 : n.id;
function hf(n) {
  try {
    for (wn = 0; wn < Ht.length; wn++) {
      const e = Ht[wn];
      e && !(e.flags & 8) && (e.flags & 4 && (e.flags &= -2), Pr(
        e,
        e.i,
        e.i ? 15 : 14
      ), e.flags & 4 || (e.flags &= -2));
    }
  } finally {
    for (; wn < Ht.length; wn++) {
      const e = Ht[wn];
      e && (e.flags &= -2);
    }
    wn = -1, Ht.length = 0, uf(), Ro = null, (Ht.length || Rs.length) && hf();
  }
}
let dn = null, ff = null;
function Co(n) {
  const e = dn;
  return dn = n, ff = n && n.type.__scopeId || null, e;
}
function df(n, e = dn, t) {
  if (!e || n._n)
    return n;
  const i = (...s) => {
    i._d && Io(-1);
    const r = Co(e), o = Wi.length;
    let a;
    try {
      a = n(...s);
    } finally {
      for (let l = Wi.length; l > o; l--) Vf();
      Co(r), i._d && Io(1);
    }
    return a;
  };
  return i._n = !0, i._c = !0, i._d = !0, i;
}
function Op(n, e) {
  if (dn === null)
    return n;
  const t = ea(dn), i = n.dirs || (n.dirs = []);
  for (let s = 0; s < e.length; s++) {
    let [r, o, a, l = ut] = e[s];
    r && (Xe(r) && (r = {
      mounted: r,
      updated: r
    }), r.deep && Jn(o), i.push({
      dir: r,
      instance: t,
      value: o,
      oldValue: void 0,
      arg: a,
      modifiers: l
    }));
  }
  return n;
}
function bi(n, e, t, i) {
  const s = n.dirs, r = e && e.dirs;
  for (let o = 0; o < s.length; o++) {
    const a = s[o];
    r && (a.oldValue = r[o].value);
    let l = a.dir[i];
    l && (ni(), mn(l, t, 8, [
      n.el,
      a,
      n,
      e
    ]), ii());
  }
}
function Bp(n, e) {
  if (Gt) {
    let t = Gt.provides;
    const i = Gt.parent && Gt.parent.provides;
    i === t && (t = Gt.provides = Object.create(i)), t[n] = e;
  }
}
function go(n, e, t = !1) {
  const i = Wf();
  if (i || Cs) {
    let s = Cs ? Cs._context.provides : i ? i.parent == null || i.ce ? i.vnode.appContext && i.vnode.appContext.provides : i.parent.provides : void 0;
    if (s && n in s)
      return s[n];
    if (arguments.length > 1)
      return t && Xe(e) ? e.call(i && i.proxy) : e;
  }
}
const zp = /* @__PURE__ */ Symbol.for("v-scx"), Hp = () => go(zp);
function vo(n, e, t) {
  return pf(n, e, t);
}
function pf(n, e, t = ut) {
  const { immediate: i, deep: s, flush: r, once: o } = t, a = Rt({}, t), l = e && i || !e && r !== "post";
  let c;
  if (yr) {
    if (r === "sync") {
      const p = Hp();
      c = p.__watcherHandles || (p.__watcherHandles = []);
    } else if (!l) {
      const p = () => {
      };
      return p.stop = Fn, p.resume = Fn, p.pause = Fn, p;
    }
  }
  const u = Gt;
  a.call = (p, v, x) => mn(p, u, v, x);
  let h = !1;
  r === "post" ? a.scheduler = (p) => {
    jt(p, u && u.suspense);
  } : r !== "sync" && (h = !0, a.scheduler = (p, v) => {
    v ? p() : dc(p);
  }), a.augmentJob = (p) => {
    e && (p.flags |= 4), h && (p.flags |= 2, u && (p.id = u.uid, p.i = u));
  };
  const f = Lp(n, e, a);
  return yr && (c ? c.push(f) : l && f()), f;
}
function Vp(n, e, t) {
  const i = this.proxy, s = xt(n) ? n.includes(".") ? mf(i, n) : () => i[n] : n.bind(i, i);
  let r;
  Xe(e) ? r = e : (r = e.handler, t = e);
  const o = Dr(this), a = pf(s, r.bind(i), t);
  return o(), a;
}
function mf(n, e) {
  const t = e.split(".");
  return () => {
    let i = n;
    for (let s = 0; s < t.length && i; s++)
      i = i[t[s]];
    return i;
  };
}
const kp = /* @__PURE__ */ Symbol("_vte"), Ko = (n) => n.__isTeleport, hn = /* @__PURE__ */ Symbol("_leaveCb"), Xs = /* @__PURE__ */ Symbol("_enterCb");
function Gp() {
  const n = {
    isMounted: !1,
    isLeaving: !1,
    isUnmounting: !1,
    leavingVNodes: /* @__PURE__ */ new Map()
  };
  return pc(() => {
    n.isMounted = !0;
  }), mc(() => {
    n.isUnmounting = !0;
  }), n;
}
const an = [Function, Array], _f = {
  mode: String,
  appear: Boolean,
  persisted: Boolean,
  // enter
  onBeforeEnter: an,
  onEnter: an,
  onAfterEnter: an,
  onEnterCancelled: an,
  // leave
  onBeforeLeave: an,
  onLeave: an,
  onAfterLeave: an,
  onLeaveCancelled: an,
  // appear
  onBeforeAppear: an,
  onAppear: an,
  onAfterAppear: an,
  onAppearCancelled: an
}, gf = (n) => {
  const e = n.subTree;
  return e.component ? gf(e.component) : e;
}, Wp = {
  name: "BaseTransition",
  props: _f,
  setup(n, { slots: e }) {
    const t = Wf(), i = Gp();
    return () => {
      const s = e.default && Mf(e.default(), !0), r = s && s.length ? vf(s) : (
        // Keep explicit default-slot conditionals on the same transition path
        // as regular v-if branches, which render a comment placeholder.
        t.subTree ? sr() : void 0
      );
      if (!r)
        return;
      const o = /* @__PURE__ */ et(n), { mode: a } = o;
      if (i.isLeaving)
        return fa(r);
      const l = Po(r);
      if (!l)
        return fa(r);
      let c = ol(
        l,
        o,
        i,
        t,
        // #11061, ensure enterHooks is fresh after clone
        (h) => c = h
      );
      l.type !== kt && xr(l, c);
      let u = t.subTree && Po(t.subTree);
      if (u && u.type !== kt && !Oi(u, l) && gf(t).type !== kt) {
        let h = ol(
          u,
          o,
          i,
          t
        );
        if (xr(u, h), a === "out-in" && l.type !== kt)
          return i.isLeaving = !0, h.afterLeave = () => {
            i.isLeaving = !1, t.job.flags & 8 || t.update(), delete h.afterLeave, u = void 0;
          }, fa(r);
        a === "in-out" && l.type !== kt ? h.delayLeave = (f, p, v) => {
          const x = xf(
            i,
            u
          );
          x[String(u.key)] = u, f[hn] = () => {
            p(), f[hn] = void 0, delete c.delayedLeave, u = void 0;
          }, c.delayedLeave = () => {
            v(), delete c.delayedLeave, u = void 0;
          };
        } : u = void 0;
      } else u && (u = void 0);
      return r;
    };
  }
};
function vf(n) {
  let e = n[0];
  if (n.length > 1) {
    for (const t of n)
      if (t.type !== kt) {
        e = t;
        break;
      }
  }
  return e;
}
const Xp = Wp;
function xf(n, e) {
  const { leavingVNodes: t } = n;
  let i = t.get(e.type);
  return i || (i = /* @__PURE__ */ Object.create(null), t.set(e.type, i)), i;
}
function ol(n, e, t, i, s) {
  const {
    appear: r,
    mode: o,
    persisted: a = !1,
    onBeforeEnter: l,
    onEnter: c,
    onAfterEnter: u,
    onEnterCancelled: h,
    onBeforeLeave: f,
    onLeave: p,
    onAfterLeave: v,
    onLeaveCancelled: x,
    onBeforeAppear: m,
    onAppear: d,
    onAfterAppear: b,
    onAppearCancelled: A
  } = e, M = String(n.key), R = xf(t, n), w = (y, S) => {
    y && mn(
      y,
      i,
      9,
      S
    );
  }, D = (y, S) => {
    const P = S[1];
    w(y, S), ze(y) ? y.every((L) => L.length <= 1) && P() : y.length <= 1 && P();
  }, U = {
    mode: o,
    persisted: a,
    beforeEnter(y) {
      let S = l;
      if (!t.isMounted)
        if (r)
          S = m || l;
        else
          return;
      y[hn] && y[hn](
        !0
        /* cancelled */
      );
      const P = R[M];
      P && Oi(n, P) && P.el[hn] && P.el[hn](), w(S, [y]);
    },
    enter(y) {
      if (R[M] === n) return;
      let S = c, P = u, L = h;
      if (!t.isMounted)
        if (r)
          S = d || c, P = b || u, L = A || h;
        else
          return;
      let V = !1;
      y[Xs] = (te) => {
        V || (V = !0, te ? w(L, [y]) : w(P, [y]), U.delayedLeave && U.delayedLeave(), y[Xs] = void 0);
      };
      const Z = y[Xs].bind(null, !1);
      S ? D(S, [y, Z]) : Z();
    },
    leave(y, S) {
      const P = String(n.key);
      if (y[Xs] && y[Xs](
        !0
        /* cancelled */
      ), t.isUnmounting)
        return S();
      w(f, [y]);
      let L = !1;
      y[hn] = (Z) => {
        L || (L = !0, S(), Z ? w(x, [y]) : w(v, [y]), y[hn] = void 0, R[P] === n && delete R[P]);
      };
      const V = y[hn].bind(null, !1);
      R[P] = n, p ? D(p, [y, V]) : V();
    },
    clone(y) {
      const S = ol(
        y,
        e,
        t,
        i,
        s
      );
      return s && s(S), S;
    }
  };
  return U;
}
function fa(n) {
  if ($o(n))
    return n = Mi(n), n.children = null, n;
}
function Po(n) {
  if (!$o(n))
    return Ko(n.type) && n.children ? vf(n.children) : n;
  if (n.component)
    return n.component.subTree;
  const { shapeFlag: e, children: t } = n;
  if (t) {
    if (e & 16)
      return t[0];
    if (e & 32 && Xe(t.default))
      return t.default();
  }
}
function xr(n, e) {
  if (n.shapeFlag & 6 && n.component) {
    n.transition = e;
    const t = n.component.subTree;
    xr(
      Ko(t.type) && Po(t) || t,
      e
    );
  } else n.shapeFlag & 128 ? (n.ssContent.transition = e.clone(n.ssContent), n.ssFallback.transition = e.clone(n.ssFallback)) : n.transition = e;
}
function Mf(n, e = !1, t) {
  let i = [], s = 0;
  for (let r = 0; r < n.length; r++) {
    let o = n[r];
    const a = t == null ? o.key : String(t) + String(o.key != null ? o.key : r);
    o.type === Vt ? (o.patchFlag & 128 && s++, i = i.concat(
      Mf(o.children, e, a)
    )) : (e || o.type !== kt) && i.push(a != null ? Mi(o, { key: a }) : o);
  }
  if (s > 1)
    for (let r = 0; r < i.length; r++)
      i[r].patchFlag = -2;
  return i;
}
function Sf(n) {
  n.ids = [n.ids[0] + n.ids[2]++ + "-", 0, 0];
}
function Kc(n, e) {
  let t;
  return !!((t = Object.getOwnPropertyDescriptor(n, e)) && !t.configurable);
}
const Do = /* @__PURE__ */ new WeakMap();
function ur(n, e, t, i, s = !1) {
  if (ze(n)) {
    n.forEach(
      (x, m) => ur(
        x,
        e && (ze(e) ? e[m] : e),
        t,
        i,
        s
      )
    );
    return;
  }
  if (hr(i) && !s) {
    i.shapeFlag & 512 && i.type.__asyncResolved && i.component.subTree.component && ur(n, e, t, i.component.subTree);
    return;
  }
  const r = i.shapeFlag & 4 ? ea(i.component) : i.el, o = s ? null : r, { i: a, r: l } = n, c = e && e.r, u = a.refs === ut ? a.refs = {} : a.refs, h = a.setupState, f = /* @__PURE__ */ et(h), p = h === ut ? Fh : (x) => Kc(u, x) ? !1 : it(f, x), v = (x, m) => !(m && Kc(u, m));
  if (c != null && c !== l) {
    if ($c(e), xt(c))
      u[c] = null, p(c) && (h[c] = null);
    else if (/* @__PURE__ */ Ut(c)) {
      const x = e;
      v(c, x.k) && (c.value = null), x.k && (u[x.k] = null);
    }
  }
  if (Xe(l))
    Pr(l, a, 12, [o, u]);
  else {
    const x = xt(l), m = /* @__PURE__ */ Ut(l);
    if (x || m) {
      const d = () => {
        if (n.f) {
          const b = x ? p(l) ? h[l] : u[l] : v() || !n.k ? l.value : u[n.k];
          if (s)
            ze(b) && ic(b, r);
          else if (ze(b))
            b.includes(r) || b.push(r);
          else if (x)
            u[l] = [r], p(l) && (h[l] = u[l]);
          else {
            const A = [r];
            v(l, n.k) && (l.value = A), n.k && (u[n.k] = A);
          }
        } else x ? (u[l] = o, p(l) && (h[l] = o)) : m && (v(l, n.k) && (l.value = o), n.k && (u[n.k] = o));
      };
      if (o) {
        const b = () => {
          d(), Do.delete(n);
        };
        b.id = -1, Do.set(n, b), jt(b, t);
      } else
        $c(n), d();
    }
  }
}
function $c(n) {
  const e = Do.get(n);
  e && (e.flags |= 8, Do.delete(n));
}
Yo().requestIdleCallback;
Yo().cancelIdleCallback;
const hr = (n) => !!n.type.__asyncLoader, $o = (n) => n.type.__isKeepAlive;
function Yp(n, e) {
  yf(n, "a", e);
}
function qp(n, e) {
  yf(n, "da", e);
}
function yf(n, e, t = Gt) {
  const i = n.__wdc || (n.__wdc = () => {
    let s = t;
    for (; s; ) {
      if (s.isDeactivated)
        return;
      s = s.parent;
    }
    return n();
  });
  if (Zo(e, i, t), t) {
    let s = t.parent;
    for (; s && s.parent; )
      $o(s.parent.vnode) && jp(i, e, t, s), s = s.parent;
  }
}
function jp(n, e, t, i) {
  const s = Zo(
    e,
    n,
    i,
    !0
    /* prepend */
  );
  Ef(() => {
    ic(i[e], s);
  }, t);
}
function Zo(n, e, t = Gt, i = !1) {
  if (t) {
    const s = t[n] || (t[n] = []), r = e.__weh || (e.__weh = (...o) => {
      ni();
      const a = Dr(t), l = mn(e, t, n, o);
      return a(), ii(), l;
    });
    return i ? s.unshift(r) : s.push(r), r;
  }
}
const ri = (n) => (e, t = Gt) => {
  (!yr || n === "sp") && Zo(n, (...i) => e(...i), t);
}, Kp = ri("bm"), pc = ri("m"), $p = ri(
  "bu"
), Zp = ri("u"), mc = ri(
  "bum"
), Ef = ri("um"), Jp = ri(
  "sp"
), Qp = ri("rtg"), em = ri("rtc");
function tm(n, e = Gt) {
  Zo("ec", n, e);
}
const nm = /* @__PURE__ */ Symbol.for("v-ndc");
function Ys(n, e, t, i) {
  let s;
  const r = t, o = ze(n);
  if (o || xt(n)) {
    const a = o && /* @__PURE__ */ Gi(n);
    let l = !1, c = !1;
    a && (l = !/* @__PURE__ */ pn(n), c = /* @__PURE__ */ si(n), n = qo(n)), s = new Array(n.length);
    for (let u = 0, h = n.length; u < h; u++)
      s[u] = e(
        l ? c ? Is(Tn(n[u])) : Tn(n[u]) : n[u],
        u,
        void 0,
        r
      );
  } else if (typeof n == "number") {
    s = new Array(n);
    for (let a = 0; a < n; a++)
      s[a] = e(a + 1, a, void 0, r);
  } else if (st(n))
    if (n[Symbol.iterator])
      s = Array.from(
        n,
        (a, l) => e(a, l, void 0, r)
      );
    else {
      const a = Object.keys(n);
      s = new Array(a.length);
      for (let l = 0, c = a.length; l < c; l++) {
        const u = a[l];
        s[l] = e(n[u], u, l, r);
      }
    }
  else
    s = [];
  return s;
}
const al = (n) => n ? Xf(n) ? ea(n) : al(n.parent) : null, fr = (
  // Move PURE marker to new line to workaround compiler discarding it
  // due to type annotation
  /* @__PURE__ */ Rt(/* @__PURE__ */ Object.create(null), {
    $: (n) => n,
    $el: (n) => n.vnode.el,
    $data: (n) => n.data,
    $props: (n) => n.props,
    $attrs: (n) => n.attrs,
    $slots: (n) => n.slots,
    $refs: (n) => n.refs,
    $parent: (n) => al(n.parent),
    $root: (n) => al(n.root),
    $host: (n) => n.ce,
    $emit: (n) => n.emit,
    $options: (n) => bf(n),
    $forceUpdate: (n) => n.f || (n.f = () => {
      dc(n.update);
    }),
    $nextTick: (n) => n.n || (n.n = Up.bind(n.proxy)),
    $watch: (n) => Vp.bind(n)
  })
), da = (n, e) => n !== ut && !n.__isScriptSetup && it(n, e), im = {
  get({ _: n }, e) {
    if (e === "__v_skip")
      return !0;
    const { ctx: t, setupState: i, data: s, props: r, accessCache: o, type: a, appContext: l } = n;
    if (e[0] !== "$") {
      const f = o[e];
      if (f !== void 0)
        switch (f) {
          case 1:
            return i[e];
          case 2:
            return s[e];
          case 4:
            return t[e];
          case 3:
            return r[e];
        }
      else {
        if (da(i, e))
          return o[e] = 1, i[e];
        if (s !== ut && it(s, e))
          return o[e] = 2, s[e];
        if (it(r, e))
          return o[e] = 3, r[e];
        if (t !== ut && it(t, e))
          return o[e] = 4, t[e];
        ll && (o[e] = 0);
      }
    }
    const c = fr[e];
    let u, h;
    if (c)
      return e === "$attrs" && It(n.attrs, "get", ""), c(n);
    if (
      // css module (injected by vue-loader)
      (u = a.__cssModules) && (u = u[e])
    )
      return u;
    if (t !== ut && it(t, e))
      return o[e] = 4, t[e];
    if (
      // global properties
      h = l.config.globalProperties, it(h, e)
    )
      return h[e];
  },
  set({ _: n }, e, t) {
    const { data: i, setupState: s, ctx: r } = n;
    return da(s, e) ? (s[e] = t, !0) : i !== ut && it(i, e) ? (i[e] = t, !0) : it(n.props, e) || e[0] === "$" && e.slice(1) in n ? !1 : (r[e] = t, !0);
  },
  has({
    _: { data: n, setupState: e, accessCache: t, ctx: i, appContext: s, props: r, type: o }
  }, a) {
    let l;
    return !!(t[a] || n !== ut && a[0] !== "$" && it(n, a) || da(e, a) || it(r, a) || it(i, a) || it(fr, a) || it(s.config.globalProperties, a) || (l = o.__cssModules) && l[a]);
  },
  defineProperty(n, e, t) {
    return t.get != null ? n._.accessCache[e] = 0 : it(t, "value") && this.set(n, e, t.value, null), Reflect.defineProperty(n, e, t);
  }
};
function Zc(n) {
  return ze(n) ? n.reduce(
    (e, t) => (e[t] = null, e),
    {}
  ) : n;
}
let ll = !0;
function sm(n) {
  const e = bf(n), t = n.proxy, i = n.ctx;
  ll = !1, e.beforeCreate && Jc(e.beforeCreate, n, "bc");
  const {
    // state
    data: s,
    computed: r,
    methods: o,
    watch: a,
    provide: l,
    inject: c,
    // lifecycle
    created: u,
    beforeMount: h,
    mounted: f,
    beforeUpdate: p,
    updated: v,
    activated: x,
    deactivated: m,
    beforeDestroy: d,
    beforeUnmount: b,
    destroyed: A,
    unmounted: M,
    render: R,
    renderTracked: w,
    renderTriggered: D,
    errorCaptured: U,
    serverPrefetch: y,
    // public API
    expose: S,
    inheritAttrs: P,
    // assets
    components: L,
    directives: V,
    filters: Z
  } = e;
  if (c && rm(c, i, null), o)
    for (const ie in o) {
      const H = o[ie];
      Xe(H) && (i[ie] = H.bind(t));
    }
  if (s) {
    const ie = s.call(t, t);
    st(ie) && (n.data = /* @__PURE__ */ gr(ie));
  }
  if (ll = !0, r)
    for (const ie in r) {
      const H = r[ie], fe = Xe(H) ? H.bind(t, t) : Xe(H.get) ? H.get.bind(t, t) : Fn, xe = !Xe(H) && Xe(H.set) ? H.set.bind(t) : Fn, me = un({
        get: fe,
        set: xe
      });
      Object.defineProperty(i, ie, {
        enumerable: !0,
        configurable: !0,
        get: () => me.value,
        set: (de) => me.value = de
      });
    }
  if (a)
    for (const ie in a)
      Tf(a[ie], i, t, ie);
  if (l) {
    const ie = Xe(l) ? l.call(t) : l;
    Reflect.ownKeys(ie).forEach((H) => {
      Bp(H, ie[H]);
    });
  }
  u && Jc(u, n, "c");
  function $(ie, H) {
    ze(H) ? H.forEach((fe) => ie(fe.bind(t))) : H && ie(H.bind(t));
  }
  if ($(Kp, h), $(pc, f), $($p, p), $(Zp, v), $(Yp, x), $(qp, m), $(tm, U), $(em, w), $(Qp, D), $(mc, b), $(Ef, M), $(Jp, y), ze(S))
    if (S.length) {
      const ie = n.exposed || (n.exposed = {});
      S.forEach((H) => {
        Object.defineProperty(ie, H, {
          get: () => t[H],
          set: (fe) => t[H] = fe,
          enumerable: !0
        });
      });
    } else n.exposed || (n.exposed = {});
  R && n.render === Fn && (n.render = R), P != null && (n.inheritAttrs = P), L && (n.components = L), V && (n.directives = V), y && Sf(n);
}
function rm(n, e, t = Fn) {
  ze(n) && (n = cl(n));
  for (const i in n) {
    const s = n[i];
    let r;
    st(s) ? "default" in s ? r = go(
      s.from || i,
      s.default,
      !0
    ) : r = go(s.from || i) : r = go(s), /* @__PURE__ */ Ut(r) ? Object.defineProperty(e, i, {
      enumerable: !0,
      configurable: !0,
      get: () => r.value,
      set: (o) => r.value = o
    }) : e[i] = r;
  }
}
function Jc(n, e, t) {
  mn(
    ze(n) ? n.map((i) => i.bind(e.proxy)) : n.bind(e.proxy),
    e,
    t
  );
}
function Tf(n, e, t, i) {
  let s = i.includes(".") ? mf(t, i) : () => t[i];
  if (xt(n)) {
    const r = e[n];
    Xe(r) && vo(s, r);
  } else if (Xe(n))
    vo(s, n.bind(t));
  else if (st(n))
    if (ze(n))
      n.forEach((r) => Tf(r, e, t, i));
    else {
      const r = Xe(n.handler) ? n.handler.bind(t) : e[n.handler];
      Xe(r) && vo(s, r, n);
    }
}
function bf(n) {
  const e = n.type, { mixins: t, extends: i } = e, {
    mixins: s,
    optionsCache: r,
    config: { optionMergeStrategies: o }
  } = n.appContext, a = r.get(e);
  let l;
  return a ? l = a : !s.length && !t && !i ? l = e : (l = {}, s.length && s.forEach(
    (c) => Lo(l, c, o, !0)
  ), Lo(l, e, o)), st(e) && r.set(e, l), l;
}
function Lo(n, e, t, i = !1) {
  const { mixins: s, extends: r } = e;
  r && Lo(n, r, t, !0), s && s.forEach(
    (o) => Lo(n, o, t, !0)
  );
  for (const o in e)
    if (!(i && o === "expose")) {
      const a = om[o] || t && t[o];
      n[o] = a ? a(n[o], e[o]) : e[o];
    }
  return n;
}
const om = {
  data: Qc,
  props: eu,
  emits: eu,
  // objects
  methods: ir,
  computed: ir,
  // lifecycle
  beforeCreate: Bt,
  created: Bt,
  beforeMount: Bt,
  mounted: Bt,
  beforeUpdate: Bt,
  updated: Bt,
  beforeDestroy: Bt,
  beforeUnmount: Bt,
  destroyed: Bt,
  unmounted: Bt,
  activated: Bt,
  deactivated: Bt,
  errorCaptured: Bt,
  serverPrefetch: Bt,
  // assets
  components: ir,
  directives: ir,
  // watch
  watch: lm,
  // provide / inject
  provide: Qc,
  inject: am
};
function Qc(n, e) {
  return e ? n ? function() {
    return Rt(
      Xe(n) ? n.call(this, this) : n,
      Xe(e) ? e.call(this, this) : e
    );
  } : e : n;
}
function am(n, e) {
  return ir(cl(n), cl(e));
}
function cl(n) {
  if (ze(n)) {
    const e = {};
    for (let t = 0; t < n.length; t++)
      e[n[t]] = n[t];
    return e;
  }
  return n;
}
function Bt(n, e) {
  return n ? [...new Set([].concat(n, e))] : e;
}
function ir(n, e) {
  return n ? Rt(/* @__PURE__ */ Object.create(null), n, e) : e;
}
function eu(n, e) {
  return n ? ze(n) && ze(e) ? [.../* @__PURE__ */ new Set([...n, ...e])] : Rt(
    /* @__PURE__ */ Object.create(null),
    Zc(n),
    Zc(e ?? {})
  ) : e;
}
function lm(n, e) {
  if (!n) return e;
  if (!e) return n;
  const t = Rt(/* @__PURE__ */ Object.create(null), n);
  for (const i in e)
    t[i] = Bt(n[i], e[i]);
  return t;
}
function Af() {
  return {
    app: null,
    config: {
      isNativeTag: Fh,
      performance: !1,
      globalProperties: {},
      optionMergeStrategies: {},
      errorHandler: void 0,
      warnHandler: void 0,
      compilerOptions: {}
    },
    mixins: [],
    components: {},
    directives: {},
    provides: /* @__PURE__ */ Object.create(null),
    optionsCache: /* @__PURE__ */ new WeakMap(),
    propsCache: /* @__PURE__ */ new WeakMap(),
    emitsCache: /* @__PURE__ */ new WeakMap()
  };
}
let cm = 0;
function um(n, e) {
  return function(i, s = null) {
    Xe(i) || (i = Rt({}, i)), s != null && !st(s) && (s = null);
    const r = Af(), o = /* @__PURE__ */ new WeakSet(), a = [];
    let l = !1;
    const c = r.app = {
      _uid: cm++,
      _component: i,
      _props: s,
      _container: null,
      _context: r,
      _instance: null,
      version: Gm,
      get config() {
        return r.config;
      },
      set config(u) {
      },
      use(u, ...h) {
        return o.has(u) || (u && Xe(u.install) ? (o.add(u), u.install(c, ...h)) : Xe(u) && (o.add(u), u(c, ...h))), c;
      },
      mixin(u) {
        return r.mixins.includes(u) || r.mixins.push(u), c;
      },
      component(u, h) {
        return h ? (r.components[u] = h, c) : r.components[u];
      },
      directive(u, h) {
        return h ? (r.directives[u] = h, c) : r.directives[u];
      },
      mount(u, h, f) {
        if (!l) {
          const p = c._ceVNode || Kt(i, s);
          return p.appContext = r, f === !0 ? f = "svg" : f === !1 && (f = void 0), n(p, u, f), l = !0, c._container = u, u.__vue_app__ = c, ea(p.component);
        }
      },
      onUnmount(u) {
        a.push(u);
      },
      unmount() {
        l && (mn(
          a,
          c._instance,
          16
        ), n(null, c._container), delete c._container.__vue_app__);
      },
      provide(u, h) {
        return r.provides[u] = h, c;
      },
      runWithContext(u) {
        const h = Cs;
        Cs = c;
        try {
          return u();
        } finally {
          Cs = h;
        }
      }
    };
    return c;
  };
}
let Cs = null;
const hm = (n, e) => e === "modelValue" || e === "model-value" ? n.modelModifiers : n[`${e}Modifiers`] || n[`${Mn(e)}Modifiers`] || n[`${Ki(e)}Modifiers`];
function fm(n, e, ...t) {
  if (n.isUnmounted) return;
  const i = n.vnode.props || ut;
  let s = t;
  const r = e.startsWith("update:"), o = r && hm(i, e.slice(7));
  o && (o.trim && (s = t.map((u) => xt(u) ? u.trim() : u)), o.number && (s = t.map($d)));
  let a, l = i[a = oa(e)] || // also try camelCase event handler (#2249)
  i[a = oa(Mn(e))];
  !l && r && (l = i[a = oa(Ki(e))]), l && mn(
    l,
    n,
    6,
    s
  );
  const c = i[a + "Once"];
  if (c) {
    if (!n.emitted)
      n.emitted = {};
    else if (n.emitted[a])
      return;
    n.emitted[a] = !0, mn(
      c,
      n,
      6,
      s
    );
  }
}
const dm = /* @__PURE__ */ new WeakMap();
function wf(n, e, t = !1) {
  const i = t ? dm : e.emitsCache, s = i.get(n);
  if (s !== void 0)
    return s;
  const r = n.emits;
  let o = {}, a = !1;
  if (!Xe(n)) {
    const l = (c) => {
      const u = wf(c, e, !0);
      u && (a = !0, Rt(o, u));
    };
    !t && e.mixins.length && e.mixins.forEach(l), n.extends && l(n.extends), n.mixins && n.mixins.forEach(l);
  }
  return !r && !a ? (st(n) && i.set(n, null), null) : (ze(r) ? r.forEach((l) => o[l] = null) : Rt(o, r), st(n) && i.set(n, o), o);
}
function Jo(n, e) {
  return !n || !Go(e) ? !1 : (e = e.slice(2), e = e === "Once" ? e : e.replace(/Once$/, ""), it(n, e[0].toLowerCase() + e.slice(1)) || it(n, Ki(e)) || it(n, e));
}
function tu(n) {
  const {
    type: e,
    vnode: t,
    proxy: i,
    withProxy: s,
    propsOptions: [r],
    slots: o,
    attrs: a,
    emit: l,
    render: c,
    renderCache: u,
    props: h,
    data: f,
    setupState: p,
    ctx: v,
    inheritAttrs: x
  } = n, m = Co(n);
  let d, b;
  try {
    if (t.shapeFlag & 4) {
      const M = s || i, R = M;
      d = Pn(
        c.call(
          R,
          M,
          u,
          h,
          p,
          f,
          v
        )
      ), b = a;
    } else {
      const M = e;
      d = Pn(
        M.length > 1 ? M(
          h,
          { attrs: a, slots: o, emit: l }
        ) : M(
          h,
          null
        )
      ), b = e.props ? a : pm(a);
    }
  } catch (M) {
    Wi.length = 0, jo(M, n, 1), d = Kt(kt);
  }
  let A = d;
  if (b && x !== !1) {
    const M = Object.keys(b), { shapeFlag: R } = A;
    M.length && R & 7 && (r && M.some(Wo) && (b = mm(
      b,
      r
    )), A = Mi(A, b, !1, !0));
  }
  if (t.dirs && (A = Mi(A, null, !1, !0), A.dirs = A.dirs ? A.dirs.concat(t.dirs) : t.dirs), t.transition) {
    const M = Ko(A.type) && Po(A) || A;
    xr(M, t.transition);
  }
  return d = A, Co(m), d;
}
const pm = (n) => {
  let e;
  for (const t in n)
    (t === "class" || t === "style" || Go(t)) && ((e || (e = {}))[t] = n[t]);
  return e;
}, mm = (n, e) => {
  const t = {};
  for (const i in n)
    (!Wo(i) || !(i.slice(9) in e)) && (t[i] = n[i]);
  return t;
};
function _m(n, e, t) {
  const { props: i, children: s, component: r } = n, { props: o, children: a, patchFlag: l } = e, c = r.emitsOptions;
  if (e.dirs || e.transition)
    return !0;
  if (t && l >= 0) {
    if (l & 1024)
      return !0;
    if (l & 16)
      return i ? nu(i, o, c) : !!o;
    if (l & 8) {
      const u = e.dynamicProps;
      for (let h = 0; h < u.length; h++) {
        const f = u[h];
        if (Rf(o, i, f) && !Jo(c, f))
          return !0;
      }
    }
  } else
    return (s || a) && (!a || !a.$stable) ? !0 : i === o ? !1 : i ? o ? nu(i, o, c) : !0 : !!o;
  return !1;
}
function nu(n, e, t) {
  const i = Object.keys(e);
  if (i.length !== Object.keys(n).length)
    return !0;
  for (let s = 0; s < i.length; s++) {
    const r = i[s];
    if (Rf(e, n, r) && !Jo(t, r))
      return !0;
  }
  return !1;
}
function Rf(n, e, t) {
  const i = n[t], s = e[t];
  return t === "style" && st(i) && st(s) ? !rc(i, s) : i !== s;
}
function gm({ vnode: n, parent: e, suspense: t }, i) {
  for (; e; ) {
    const s = e.subTree;
    if (s.suspense && s.suspense.activeBranch === n && (s.suspense.vnode.el = s.el = i, n = s), s === n)
      (n = e.vnode).el = i, e = e.parent;
    else
      break;
  }
  t && t.activeBranch === n && (t.vnode.el = i);
}
const Cf = {}, Pf = () => Object.create(Cf), Df = (n) => Object.getPrototypeOf(n) === Cf;
function vm(n, e, t, i = !1) {
  const s = {}, r = Pf();
  n.propsDefaults = /* @__PURE__ */ Object.create(null), Lf(n, e, s, r);
  for (const o in n.propsOptions[0])
    o in s || (s[o] = void 0);
  t ? n.props = i ? s : /* @__PURE__ */ Tp(s) : n.type.props ? n.props = s : n.props = r, n.attrs = r;
}
function xm(n, e, t, i) {
  const {
    props: s,
    attrs: r,
    vnode: { patchFlag: o }
  } = n, a = /* @__PURE__ */ et(s), [l] = n.propsOptions;
  let c = !1;
  if (
    // always force full diff in dev
    // - #1942 if hmr is enabled with sfc component
    // - vite#872 non-sfc component used by sfc component
    (i || o > 0) && !(o & 16)
  ) {
    if (o & 8) {
      const u = n.vnode.dynamicProps;
      for (let h = 0; h < u.length; h++) {
        let f = u[h];
        if (Jo(n.emitsOptions, f))
          continue;
        const p = e[f];
        if (l)
          if (it(r, f))
            p !== r[f] && (r[f] = p, c = !0);
          else {
            const v = Mn(f);
            s[v] = ul(
              l,
              a,
              v,
              p,
              n,
              !1
            );
          }
        else
          p !== r[f] && (r[f] = p, c = !0);
      }
    }
  } else {
    Lf(n, e, s, r) && (c = !0);
    let u;
    for (const h in a)
      (!e || // for camelCase
      !it(e, h) && // it's possible the original props was passed in as kebab-case
      // and converted to camelCase (#955)
      ((u = Ki(h)) === h || !it(e, u))) && (l ? t && // for camelCase
      (t[h] !== void 0 || // for kebab-case
      t[u] !== void 0) && (s[h] = ul(
        l,
        a,
        h,
        void 0,
        n,
        !0
      )) : delete s[h]);
    if (r !== a)
      for (const h in r)
        (!e || !it(e, h)) && (delete r[h], c = !0);
  }
  c && Zn(n.attrs, "set", "");
}
function Lf(n, e, t, i) {
  const [s, r] = n.propsOptions;
  let o = !1, a;
  if (e)
    for (let l in e) {
      if (ar(l))
        continue;
      const c = e[l];
      let u;
      s && it(s, u = Mn(l)) ? !r || !r.includes(u) ? t[u] = c : (a || (a = {}))[u] = c : Jo(n.emitsOptions, l) || (!(l in i) || c !== i[l]) && (i[l] = c, o = !0);
    }
  if (r) {
    const l = /* @__PURE__ */ et(t), c = a || ut;
    for (let u = 0; u < r.length; u++) {
      const h = r[u];
      t[h] = ul(
        s,
        l,
        h,
        c[h],
        n,
        !it(c, h)
      );
    }
  }
  return o;
}
function ul(n, e, t, i, s, r) {
  const o = n[t];
  if (o != null) {
    const a = it(o, "default");
    if (a && i === void 0) {
      const l = o.default;
      if (o.type !== Function && !o.skipFactory && Xe(l)) {
        const { propsDefaults: c } = s;
        if (t in c)
          i = c[t];
        else {
          const u = Dr(s);
          i = c[t] = l.call(
            null,
            e
          ), u();
        }
      } else
        i = l;
      s.ce && s.ce._setProp(t, i);
    }
    o[
      0
      /* shouldCast */
    ] && (r && !a ? i = !1 : o[
      1
      /* shouldCastTrue */
    ] && (i === "" || i === Ki(t)) && (i = !0));
  }
  return i;
}
const Mm = /* @__PURE__ */ new WeakMap();
function If(n, e, t = !1) {
  const i = t ? Mm : e.propsCache, s = i.get(n);
  if (s)
    return s;
  const r = n.props, o = {}, a = [];
  let l = !1;
  if (!Xe(n)) {
    const u = (h) => {
      l = !0;
      const [f, p] = If(h, e, !0);
      Rt(o, f), p && a.push(...p);
    };
    !t && e.mixins.length && e.mixins.forEach(u), n.extends && u(n.extends), n.mixins && n.mixins.forEach(u);
  }
  if (!r && !l)
    return st(n) && i.set(n, As), As;
  if (ze(r))
    for (let u = 0; u < r.length; u++) {
      const h = Mn(r[u]);
      iu(h) && (o[h] = ut);
    }
  else if (r)
    for (const u in r) {
      const h = Mn(u);
      if (iu(h)) {
        const f = r[u], p = o[h] = ze(f) || Xe(f) ? { type: f } : Rt({}, f), v = p.type;
        let x = !1, m = !0;
        if (ze(v))
          for (let d = 0; d < v.length; ++d) {
            const b = v[d], A = Xe(b) && b.name;
            if (A === "Boolean") {
              x = !0;
              break;
            } else A === "String" && (m = !1);
          }
        else
          x = Xe(v) && v.name === "Boolean";
        p[
          0
          /* shouldCast */
        ] = x, p[
          1
          /* shouldCastTrue */
        ] = m, (x || it(p, "default")) && a.push(h);
      }
    }
  const c = [o, a];
  return st(n) && i.set(n, c), c;
}
function iu(n) {
  return n[0] !== "$" && !ar(n);
}
const _c = (n) => n === "_" || n === "_ctx" || n === "$stable", gc = (n) => ze(n) ? n.map(Pn) : [Pn(n)], Sm = (n, e, t) => {
  if (e._n)
    return e;
  const i = df((...s) => gc(e(...s)), t);
  return i._c = !1, i;
}, Uf = (n, e, t) => {
  const i = n._ctx;
  for (const s in n) {
    if (_c(s)) continue;
    const r = n[s];
    if (Xe(r))
      e[s] = Sm(s, r, i);
    else if (r != null) {
      const o = gc(r);
      e[s] = () => o;
    }
  }
}, Nf = (n, e) => {
  const t = gc(e);
  n.slots.default = () => t;
}, Ff = (n, e, t) => {
  for (const i in e)
    (t || !_c(i)) && (n[i] = e[i]);
}, ym = (n, e, t) => {
  const i = n.slots = Pf();
  if (n.vnode.shapeFlag & 32) {
    const s = e._;
    s ? (Ff(i, e, t), t && kh(i, "_", s, !0)) : Uf(e, i);
  } else e && Nf(n, e);
}, Em = (n, e, t) => {
  const { vnode: i, slots: s } = n;
  let r = !0, o = ut;
  if (i.shapeFlag & 32) {
    const a = e._;
    a ? t && a === 1 ? r = !1 : Ff(s, e, t) : (r = !e.$stable, Uf(e, s)), o = e;
  } else e && (Nf(n, e), o = { default: 1 });
  if (r)
    for (const a in s)
      !_c(a) && o[a] == null && delete s[a];
}, jt = Rm;
function Tm(n) {
  return bm(n);
}
function bm(n, e) {
  const t = Yo();
  t.__VUE__ = !0;
  const {
    insert: i,
    remove: s,
    patchProp: r,
    createElement: o,
    createText: a,
    createComment: l,
    setText: c,
    setElementText: u,
    parentNode: h,
    nextSibling: f,
    setScopeId: p = Fn,
    insertStaticContent: v
  } = n, x = (C, g, W, j = null, X = null, z = null, ae = void 0, q = null, Q = !!g.dynamicChildren) => {
    if (C === g)
      return;
    C && !Oi(C, g) && (j = re(C), de(C, X, z, !0), C = null), g.patchFlag === -2 && (Q = !1, g.dynamicChildren = null);
    const { type: ee, ref: Se, shapeFlag: E } = g;
    switch (ee) {
      case Qo:
        m(C, g, W, j);
        break;
      case kt:
        d(C, g, W, j);
        break;
      case ma:
        C == null && b(g, W, j, ae);
        break;
      case Vt:
        L(
          C,
          g,
          W,
          j,
          X,
          z,
          ae,
          q,
          Q
        );
        break;
      default:
        E & 1 ? R(
          C,
          g,
          W,
          j,
          X,
          z,
          ae,
          q,
          Q
        ) : E & 6 ? V(
          C,
          g,
          W,
          j,
          X,
          z,
          ae,
          q,
          Q
        ) : (E & 64 || E & 128) && ee.process(
          C,
          g,
          W,
          j,
          X,
          z,
          ae,
          q,
          Q,
          Pe
        );
    }
    Se != null && X ? ur(Se, C && C.ref, z, g || C, !g) : Se == null && C && C.ref != null && ur(C.ref, null, z, C, !0);
  }, m = (C, g, W, j) => {
    if (C == null)
      i(
        g.el = a(g.children),
        W,
        j
      );
    else {
      const X = g.el = C.el;
      g.children !== C.children && c(X, g.children);
    }
  }, d = (C, g, W, j) => {
    C == null ? i(
      g.el = l(g.children || ""),
      W,
      j
    ) : g.el = C.el;
  }, b = (C, g, W, j) => {
    [C.el, C.anchor] = v(
      C.children,
      g,
      W,
      j,
      C.el,
      C.anchor
    );
  }, A = ({ el: C, anchor: g }, W, j) => {
    let X;
    for (; C && C !== g; )
      X = f(C), i(C, W, j), C = X;
    i(g, W, j);
  }, M = ({ el: C, anchor: g }) => {
    let W;
    for (; C && C !== g; )
      W = f(C), s(C), C = W;
    s(g);
  }, R = (C, g, W, j, X, z, ae, q, Q) => {
    if (g.type === "svg" ? ae = "svg" : g.type === "math" && (ae = "mathml"), C == null)
      w(
        g,
        W,
        j,
        X,
        z,
        ae,
        q,
        Q
      );
    else {
      const ee = C.el && C.el._isVueCE ? C.el : null;
      try {
        ee && ee._beginPatch(), y(
          C,
          g,
          X,
          z,
          ae,
          q,
          Q
        );
      } finally {
        ee && ee._endPatch();
      }
    }
  }, w = (C, g, W, j, X, z, ae, q) => {
    let Q, ee;
    const { props: Se, shapeFlag: E, transition: _, dirs: I } = C;
    if (Q = C.el = o(
      C.type,
      z,
      Se && Se.is,
      Se
    ), E & 8 ? u(Q, C.children) : E & 16 && U(
      C.children,
      Q,
      null,
      j,
      X,
      pa(C, z),
      ae,
      q
    ), I && bi(C, null, j, "created"), D(Q, C, C.scopeId, ae, j), Se) {
      for (const J in Se)
        J !== "value" && !ar(J) && r(Q, J, null, Se[J], z, j);
      "value" in Se && r(Q, "value", null, Se.value, z), (ee = Se.onVnodeBeforeMount) && An(ee, j, C);
    }
    I && bi(C, null, j, "beforeMount");
    const k = Am(X, _);
    k && _.beforeEnter(Q), i(Q, g, W), ((ee = Se && Se.onVnodeMounted) || k || I) && jt(() => {
      ee && An(ee, j, C), k && _.enter(Q), I && bi(C, null, j, "mounted");
    }, X);
  }, D = (C, g, W, j, X) => {
    if (W && p(C, W), j)
      for (let z = 0; z < j.length; z++)
        p(C, j[z]);
    if (X) {
      let z = X.subTree;
      if (g === z || Hf(z.type) && (z.ssContent === g || z.ssFallback === g)) {
        const ae = X.vnode;
        D(
          C,
          ae,
          ae.scopeId,
          ae.slotScopeIds,
          X.parent
        );
      }
    }
  }, U = (C, g, W, j, X, z, ae, q, Q = 0) => {
    for (let ee = Q; ee < C.length; ee++) {
      const Se = C[ee] = q ? $n(C[ee]) : Pn(C[ee]);
      x(
        null,
        Se,
        g,
        W,
        j,
        X,
        z,
        ae,
        q
      );
    }
  }, y = (C, g, W, j, X, z, ae) => {
    const q = g.el = C.el;
    let { patchFlag: Q, dynamicChildren: ee, dirs: Se } = g;
    Q |= C.patchFlag & 16;
    const E = C.props || ut, _ = g.props || ut;
    let I;
    if (W && Ai(W, !1), (I = _.onVnodeBeforeUpdate) && An(I, W, g, C), Se && bi(g, C, W, "beforeUpdate"), W && Ai(W, !0), // #6385 the old vnode may be a user-wrapped non-isomorphic block
    // Force full diff when block metadata is unstable.
    ee && (!C.dynamicChildren || C.dynamicChildren.length !== ee.length) && (Q = 0, ae = !1, ee = null), (E.innerHTML && _.innerHTML == null || E.textContent && _.textContent == null) && u(q, ""), ee ? S(
      C.dynamicChildren,
      ee,
      q,
      W,
      j,
      pa(g, X),
      z
    ) : ae || H(
      C,
      g,
      q,
      null,
      W,
      j,
      pa(g, X),
      z,
      !1
    ), Q > 0) {
      if (Q & 16)
        P(q, E, _, W, X);
      else if (Q & 2 && E.class !== _.class && r(q, "class", null, _.class, X), Q & 4 && r(q, "style", E.style, _.style, X), Q & 8) {
        const k = g.dynamicProps;
        for (let J = 0; J < k.length; J++) {
          const G = k[J], _e = E[G], oe = _[G];
          (oe !== _e || G === "value") && r(q, G, _e, oe, X, W);
        }
      }
      Q & 1 && C.children !== g.children && u(q, g.children);
    } else !ae && ee == null && P(q, E, _, W, X);
    ((I = _.onVnodeUpdated) || Se) && jt(() => {
      I && An(I, W, g, C), Se && bi(g, C, W, "updated");
    }, j);
  }, S = (C, g, W, j, X, z, ae) => {
    for (let q = 0; q < g.length; q++) {
      const Q = C[q], ee = g[q], Se = (
        // oldVNode may be an errored async setup() component inside Suspense
        // which will not have a mounted element
        Q.el && // - In the case of a Fragment, we need to provide the actual parent
        // of the Fragment itself so it can move its children.
        (Q.type === Vt || // - In the case of different nodes, there is going to be a replacement
        // which also requires the correct parent container
        !Oi(Q, ee) || // - In the case of a component, it could contain anything.
        Q.shapeFlag & 198) ? h(Q.el) : (
          // In other cases, the parent container is not actually used so we
          // just pass the block element here to avoid a DOM parentNode call.
          W
        )
      );
      x(
        Q,
        ee,
        Se,
        null,
        j,
        X,
        z,
        ae,
        !0
      );
    }
  }, P = (C, g, W, j, X) => {
    if (g !== W) {
      if (g !== ut)
        for (const z in g)
          !ar(z) && !(z in W) && r(
            C,
            z,
            g[z],
            null,
            X,
            j
          );
      for (const z in W) {
        if (ar(z)) continue;
        const ae = W[z], q = g[z];
        ae !== q && z !== "value" && r(C, z, q, ae, X, j);
      }
      "value" in W && r(C, "value", g.value, W.value, X);
    }
  }, L = (C, g, W, j, X, z, ae, q, Q) => {
    const ee = g.el = C ? C.el : a(""), Se = g.anchor = C ? C.anchor : a("");
    let { patchFlag: E, dynamicChildren: _, slotScopeIds: I } = g;
    I && (q = q ? q.concat(I) : I), C == null ? (i(ee, W, j), i(Se, W, j), U(
      // #10007
      // such fragment like `<></>` will be compiled into
      // a fragment which doesn't have a children.
      // In this case fallback to an empty array
      g.children || [],
      W,
      Se,
      X,
      z,
      ae,
      q,
      Q
    )) : E > 0 && E & 64 && _ && // #2715 the previous fragment could've been a BAILed one as a result
    // of renderSlot() with no valid children
    C.dynamicChildren && C.dynamicChildren.length === _.length ? (S(
      C.dynamicChildren,
      _,
      W,
      X,
      z,
      ae,
      q
    ), // #2080 if the stable fragment has a key, it's a <template v-for> that may
    //  get moved around. Make sure all root level vnodes inherit el.
    // #2134 or if it's a component root, it may also get moved around
    // as the component is being moved.
    (g.key != null || X && g === X.subTree) && Of(
      C,
      g,
      !0
      /* shallow */
    )) : H(
      C,
      g,
      W,
      Se,
      X,
      z,
      ae,
      q,
      Q
    );
  }, V = (C, g, W, j, X, z, ae, q, Q) => {
    g.slotScopeIds = q, C == null ? g.shapeFlag & 512 ? X.ctx.activate(
      g,
      W,
      j,
      ae,
      Q
    ) : Z(
      g,
      W,
      j,
      X,
      z,
      ae,
      Q
    ) : te(C, g, Q);
  }, Z = (C, g, W, j, X, z, ae) => {
    const q = C.component = Fm(
      C,
      j,
      X
    );
    if ($o(C) && (q.ctx.renderer = Pe), Om(q, !1, ae), q.asyncDep) {
      if (X && X.registerDep(q, $, ae), !C.el) {
        const Q = q.subTree = Kt(kt);
        d(null, Q, g, W), C.placeholder = Q.el;
      }
    } else
      $(
        q,
        C,
        g,
        W,
        X,
        z,
        ae
      );
  }, te = (C, g, W) => {
    const j = g.component = C.component;
    if (_m(C, g, W))
      if (j.asyncDep && !j.asyncResolved) {
        ie(j, g, W);
        return;
      } else
        j.next = g, j.update();
    else
      g.el = C.el, j.vnode = g;
  }, $ = (C, g, W, j, X, z, ae) => {
    const q = () => {
      if (C.isMounted) {
        let { next: E, bu: _, u: I, parent: k, vnode: J } = C;
        {
          const Te = Bf(C);
          if (Te) {
            E && (E.el = J.el, ie(C, E, ae)), Te.asyncDep.then(() => {
              jt(() => {
                C.isUnmounted || ee();
              }, X);
            });
            return;
          }
        }
        let G = E, _e;
        Ai(C, !1), E ? (E.el = J.el, ie(C, E, ae)) : E = J, _ && aa(_), (_e = E.props && E.props.onVnodeBeforeUpdate) && An(_e, k, E, J), Ai(C, !0);
        const oe = tu(C), Ee = C.subTree;
        C.subTree = oe, x(
          Ee,
          oe,
          // parent may have changed if it's in a teleport
          h(Ee.el),
          // anchor may have changed if it's in a fragment
          re(Ee),
          C,
          X,
          z
        ), E.el = oe.el, G === null && gm(C, oe.el), I && jt(I, X), (_e = E.props && E.props.onVnodeUpdated) && jt(
          () => An(_e, k, E, J),
          X
        );
      } else {
        let E;
        const { el: _, props: I } = g, { bm: k, m: J, parent: G, root: _e, type: oe } = C, Ee = hr(g);
        Ai(C, !1), k && aa(k), !Ee && (E = I && I.onVnodeBeforeMount) && An(E, G, g), Ai(C, !0);
        {
          _e.ce && _e.ce._hasShadowRoot() && _e.ce._injectChildStyle(
            oe,
            C.parent ? C.parent.type : void 0
          );
          const Te = C.subTree = tu(C);
          x(
            null,
            Te,
            W,
            j,
            C,
            X,
            z
          ), g.el = Te.el;
        }
        if (J && jt(J, X), !Ee && (E = I && I.onVnodeMounted)) {
          const Te = g;
          jt(
            () => An(E, G, Te),
            X
          );
        }
        (g.shapeFlag & 256 || G && hr(G.vnode) && G.vnode.shapeFlag & 256) && C.a && jt(C.a, X), C.isMounted = !0, g = W = j = null;
      }
    };
    C.scope.on();
    const Q = C.effect = new Yh(q);
    C.scope.off();
    const ee = C.update = Q.run.bind(Q), Se = C.job = Q.runIfDirty.bind(Q);
    Se.i = C, Se.id = C.uid, Q.scheduler = () => dc(Se), Ai(C, !0), ee();
  }, ie = (C, g, W) => {
    g.component = C;
    const j = C.vnode.props;
    C.vnode = g, C.next = null, xm(C, g.props, j, W), Em(C, g.children, W), ni(), jc(C), ii();
  }, H = (C, g, W, j, X, z, ae, q, Q = !1) => {
    const ee = C && C.children, Se = C ? C.shapeFlag : 0, E = g.children, { patchFlag: _, shapeFlag: I } = g;
    if (_ > 0) {
      if (_ & 128) {
        xe(
          ee,
          E,
          W,
          j,
          X,
          z,
          ae,
          q,
          Q
        );
        return;
      } else if (_ & 256) {
        fe(
          ee,
          E,
          W,
          j,
          X,
          z,
          ae,
          q,
          Q
        );
        return;
      }
    }
    I & 8 ? (Se & 16 && ne(ee, X, z), E !== ee && u(W, E)) : Se & 16 ? I & 16 ? xe(
      ee,
      E,
      W,
      j,
      X,
      z,
      ae,
      q,
      Q
    ) : ne(ee, X, z, !0) : (Se & 8 && u(W, ""), I & 16 && U(
      E,
      W,
      j,
      X,
      z,
      ae,
      q,
      Q
    ));
  }, fe = (C, g, W, j, X, z, ae, q, Q) => {
    C = C || As, g = g || As;
    const ee = C.length, Se = g.length, E = Math.min(ee, Se);
    let _;
    for (_ = 0; _ < E; _++) {
      const I = g[_] = Q ? $n(g[_]) : Pn(g[_]);
      x(
        C[_],
        I,
        W,
        null,
        X,
        z,
        ae,
        q,
        Q
      );
    }
    ee > Se ? ne(
      C,
      X,
      z,
      !0,
      !1,
      E
    ) : U(
      g,
      W,
      j,
      X,
      z,
      ae,
      q,
      Q,
      E
    );
  }, xe = (C, g, W, j, X, z, ae, q, Q) => {
    let ee = 0;
    const Se = g.length;
    let E = C.length - 1, _ = Se - 1;
    for (; ee <= E && ee <= _; ) {
      const I = C[ee], k = g[ee] = Q ? $n(g[ee]) : Pn(g[ee]);
      if (Oi(I, k))
        x(
          I,
          k,
          W,
          null,
          X,
          z,
          ae,
          q,
          Q
        );
      else
        break;
      ee++;
    }
    for (; ee <= E && ee <= _; ) {
      const I = C[E], k = g[_] = Q ? $n(g[_]) : Pn(g[_]);
      if (Oi(I, k))
        x(
          I,
          k,
          W,
          null,
          X,
          z,
          ae,
          q,
          Q
        );
      else
        break;
      E--, _--;
    }
    if (ee > E) {
      if (ee <= _) {
        const I = _ + 1, k = I < Se ? g[I].el : j;
        for (; ee <= _; )
          x(
            null,
            g[ee] = Q ? $n(g[ee]) : Pn(g[ee]),
            W,
            k,
            X,
            z,
            ae,
            q,
            Q
          ), ee++;
      }
    } else if (ee > _)
      for (; ee <= E; )
        de(C[ee], X, z, !0), ee++;
    else {
      const I = ee, k = ee, J = /* @__PURE__ */ new Map();
      for (ee = k; ee <= _; ee++) {
        const Ce = g[ee] = Q ? $n(g[ee]) : Pn(g[ee]);
        Ce.key != null && J.set(Ce.key, ee);
      }
      let G, _e = 0;
      const oe = _ - k + 1;
      let Ee = !1, Te = 0;
      const le = new Array(oe);
      for (ee = 0; ee < oe; ee++) le[ee] = 0;
      for (ee = I; ee <= E; ee++) {
        const Ce = C[ee];
        if (_e >= oe) {
          de(Ce, X, z, !0);
          continue;
        }
        let be;
        if (Ce.key != null)
          be = J.get(Ce.key);
        else
          for (G = k; G <= _; G++)
            if (le[G - k] === 0 && Oi(Ce, g[G])) {
              be = G;
              break;
            }
        be === void 0 ? de(Ce, X, z, !0) : (le[be - k] = ee + 1, be >= Te ? Te = be : Ee = !0, x(
          Ce,
          g[be],
          W,
          null,
          X,
          z,
          ae,
          q,
          Q
        ), _e++);
      }
      const Me = Ee ? wm(le) : As;
      for (G = Me.length - 1, ee = oe - 1; ee >= 0; ee--) {
        const Ce = k + ee, be = g[Ce], ge = g[Ce + 1], ke = Ce + 1 < Se ? (
          // #13559, #14173 fallback to el placeholder for unresolved async component
          ge.el || zf(ge)
        ) : j;
        le[ee] === 0 ? x(
          null,
          be,
          W,
          ke,
          X,
          z,
          ae,
          q,
          Q
        ) : Ee && (G < 0 || ee !== Me[G] ? me(be, W, ke, 2) : G--);
      }
    }
  }, me = (C, g, W, j, X = null) => {
    const { el: z, type: ae, transition: q, children: Q, shapeFlag: ee } = C;
    if (ee & 6) {
      me(C.component.subTree, g, W, j);
      return;
    }
    if (ee & 128) {
      C.suspense.move(g, W, j);
      return;
    }
    if (ee & 64) {
      ae.move(C, g, W, Pe);
      return;
    }
    if (ae === Vt) {
      i(z, g, W);
      for (let E = 0; E < Q.length; E++)
        me(Q[E], g, W, j);
      i(C.anchor, g, W);
      return;
    }
    if (ae === ma) {
      A(C, g, W);
      return;
    }
    if (j !== 2 && ee & 1 && q)
      if (j === 0)
        q.persisted && !z[hn] ? i(z, g, W) : (q.beforeEnter(z), i(z, g, W), jt(() => q.enter(z), X));
      else {
        const { leave: E, delayLeave: _, afterLeave: I } = q, k = () => {
          C.ctx.isUnmounted ? s(z) : i(z, g, W);
        }, J = () => {
          const G = z._isLeaving || !!z[hn];
          z._isLeaving && z[hn](
            !0
            /* cancelled */
          ), q.persisted && !G ? k() : E(z, () => {
            k(), I && I();
          });
        };
        _ ? _(z, k, J) : J();
      }
    else
      i(z, g, W);
  }, de = (C, g, W, j = !1, X = !1) => {
    const {
      type: z,
      props: ae,
      ref: q,
      children: Q,
      dynamicChildren: ee,
      shapeFlag: Se,
      patchFlag: E,
      dirs: _,
      cacheIndex: I,
      memo: k
    } = C;
    if (E === -2 && (X = !1), q != null && (ni(), ur(q, null, W, C, !0), ii()), I != null && (g.renderCache[I] = void 0), Se & 256) {
      g.ctx.deactivate(C);
      return;
    }
    const J = Se & 1 && _, G = !hr(C);
    let _e;
    if (G && (_e = ae && ae.onVnodeBeforeUnmount) && An(_e, g, C), Se & 6)
      Ze(C.component, W, j);
    else {
      if (Se & 128) {
        C.suspense.unmount(W, j);
        return;
      }
      J && bi(C, null, g, "beforeUnmount"), Se & 64 ? C.type.remove(
        C,
        g,
        W,
        Pe,
        j
      ) : ee && // #5154
      // when v-once is used inside a block, setBlockTracking(-1) marks the
      // parent block with hasOnce: true
      // so that it doesn't take the fast path during unmount - otherwise
      // components nested in v-once are never unmounted.
      !ee.hasOnce && // #1153: fast path should not be taken for non-stable (v-for) fragments
      (z !== Vt || E > 0 && E & 64) ? ne(
        ee,
        g,
        W,
        !1,
        !0
      ) : (z === Vt && E & 384 || !X && Se & 16) && ne(Q, g, W), j && Le(C);
    }
    const oe = k != null && I == null;
    (G && (_e = ae && ae.onVnodeUnmounted) || J || oe) && jt(() => {
      _e && An(_e, g, C), J && bi(C, null, g, "unmounted"), oe && (C.el = null);
    }, W);
  }, Le = (C) => {
    const { type: g, el: W, anchor: j, transition: X } = C;
    if (g === Vt) {
      tt(W, j);
      return;
    }
    if (g === ma) {
      M(C);
      return;
    }
    const z = () => {
      s(W), X && !X.persisted && X.afterLeave && X.afterLeave();
    };
    if (C.shapeFlag & 1 && X && !X.persisted) {
      const { leave: ae, delayLeave: q } = X, Q = () => ae(W, z);
      q ? q(C.el, z, Q) : Q();
    } else
      z();
  }, tt = (C, g) => {
    let W;
    for (; C !== g; )
      W = f(C), s(C), C = W;
    s(g);
  }, Ze = (C, g, W) => {
    const { bum: j, scope: X, job: z, subTree: ae, um: q, m: Q, a: ee } = C;
    su(Q), su(ee), j && aa(j), X.stop(), z && (z.flags |= 8, de(ae, C, g, W)), q && jt(q, g), jt(() => {
      C.isUnmounted = !0;
    }, g);
  }, ne = (C, g, W, j = !1, X = !1, z = 0) => {
    for (let ae = z; ae < C.length; ae++)
      de(C[ae], g, W, j, X);
  }, re = (C) => {
    if (C.shapeFlag & 6)
      return re(C.component.subTree);
    if (C.shapeFlag & 128)
      return C.suspense.next();
    const g = f(C.anchor || C.el), W = g && g[kp];
    return W ? f(W) : g;
  };
  let Ae = !1;
  const Oe = (C, g, W) => {
    let j;
    C == null ? g._vnode && (de(g._vnode, null, null, !0), j = g._vnode.component) : x(
      g._vnode || null,
      C,
      g,
      null,
      null,
      null,
      W
    ), g._vnode = C, Ae || (Ae = !0, jc(j), uf(), Ae = !1);
  }, Pe = {
    p: x,
    um: de,
    m: me,
    r: Le,
    mt: Z,
    mc: U,
    pc: H,
    pbc: S,
    n: re,
    o: n
  };
  return {
    render: Oe,
    hydrate: void 0,
    createApp: um(Oe)
  };
}
function pa({ type: n, props: e }, t) {
  return t === "svg" && n === "foreignObject" || t === "mathml" && n === "annotation-xml" && e && e.encoding && e.encoding.includes("html") ? void 0 : t;
}
function Ai({ effect: n, job: e }, t) {
  t ? (n.flags |= 32, e.flags |= 4) : (n.flags &= -33, e.flags &= -5);
}
function Am(n, e) {
  return (!n || n && !n.pendingBranch) && e && !e.persisted;
}
function Of(n, e, t = !1) {
  const i = n.children, s = e.children;
  if (ze(i) && ze(s))
    for (let r = 0; r < i.length; r++) {
      const o = i[r];
      let a = s[r];
      a.shapeFlag & 1 && !a.dynamicChildren && ((a.patchFlag <= 0 || a.patchFlag === 32) && (a = s[r] = $n(s[r]), a.el = o.el), !t && a.patchFlag !== -2 && Of(o, a)), a.type === Qo && (a.patchFlag === -1 && (a = s[r] = $n(a)), a.el = o.el), a.type === kt && !a.el && (a.el = o.el);
    }
}
function wm(n) {
  const e = n.slice(), t = [0];
  let i, s, r, o, a;
  const l = n.length;
  for (i = 0; i < l; i++) {
    const c = n[i];
    if (c !== 0) {
      if (s = t[t.length - 1], n[s] < c) {
        e[i] = s, t.push(i);
        continue;
      }
      for (r = 0, o = t.length - 1; r < o; )
        a = r + o >> 1, n[t[a]] < c ? r = a + 1 : o = a;
      c < n[t[r]] && (r > 0 && (e[i] = t[r - 1]), t[r] = i);
    }
  }
  for (r = t.length, o = t[r - 1]; r-- > 0; )
    t[r] = o, o = e[o];
  return t;
}
function Bf(n) {
  const e = n.subTree.component;
  if (e)
    return e.asyncDep && !e.asyncResolved ? e : Bf(e);
}
function su(n) {
  if (n)
    for (let e = 0; e < n.length; e++)
      n[e].flags |= 8;
}
function zf(n) {
  if (n.placeholder)
    return n.placeholder;
  const e = n.component;
  return e ? zf(e.subTree) : null;
}
const Hf = (n) => n.__isSuspense;
function Rm(n, e) {
  e && e.pendingBranch ? ze(n) ? e.effects.push(...n) : e.effects.push(n) : Fp(n);
}
const Vt = /* @__PURE__ */ Symbol.for("v-fgt"), Qo = /* @__PURE__ */ Symbol.for("v-txt"), kt = /* @__PURE__ */ Symbol.for("v-cmt"), ma = /* @__PURE__ */ Symbol.for("v-stc"), Wi = [];
let sn = null;
function Lt(n = !1) {
  Wi.push(sn = n ? null : []);
}
function Vf() {
  Wi.pop(), sn = Wi[Wi.length - 1] || null;
}
let Mr = 1;
function Io(n, e = !1) {
  Mr += n, n < 0 && sn && e && (sn.hasOnce = !0);
}
function kf(n) {
  return n.dynamicChildren = Mr > 0 ? sn || As : null, Vf(), Mr > 0 && sn && sn.push(n), n;
}
function Ot(n, e, t, i, s, r) {
  return kf(
    Be(
      n,
      e,
      t,
      i,
      s,
      r,
      !0
    )
  );
}
function Cm(n, e, t, i, s) {
  return kf(
    Kt(
      n,
      e,
      t,
      i,
      s,
      !0
    )
  );
}
function Uo(n) {
  return n ? n.__v_isVNode === !0 : !1;
}
function Oi(n, e) {
  return n.type === e.type && n.key === e.key;
}
const Gf = ({ key: n }) => n ?? null, xo = ({
  ref: n,
  ref_key: e,
  ref_for: t
}) => (typeof n == "number" && (n = "" + n), n != null ? xt(n) || /* @__PURE__ */ Ut(n) || Xe(n) ? { i: dn, r: n, k: e, f: !!t } : n : null);
function Be(n, e = null, t = null, i = 0, s = null, r = n === Vt ? 0 : 1, o = !1, a = !1) {
  const l = {
    __v_isVNode: !0,
    __v_skip: !0,
    type: n,
    props: e,
    key: e && Gf(e),
    ref: e && xo(e),
    scopeId: ff,
    slotScopeIds: null,
    children: t,
    component: null,
    suspense: null,
    ssContent: null,
    ssFallback: null,
    dirs: null,
    transition: null,
    el: null,
    anchor: null,
    target: null,
    targetStart: null,
    targetAnchor: null,
    staticCount: 0,
    shapeFlag: r,
    patchFlag: i,
    dynamicProps: s,
    dynamicChildren: null,
    appContext: null,
    ctx: dn
  };
  return a ? (No(l, t), r & 128 && n.normalize(l)) : t && (l.shapeFlag |= xt(t) ? 8 : 16), Mr > 0 && // avoid a block node from tracking itself
  !o && // has current parent block
  sn && // presence of a patch flag indicates this node needs patching on updates.
  // component nodes also should always be patched, because even if the
  // component doesn't need to update, it needs to persist the instance on to
  // the next vnode so that it can be properly unmounted later.
  (l.patchFlag > 0 || r & 6) && // the EVENTS flag is only for hydration and if it is the only flag, the
  // vnode should not be considered dynamic due to handler caching.
  l.patchFlag !== 32 && sn.push(l), l;
}
const Kt = Pm;
function Pm(n, e = null, t = null, i = 0, s = null, r = !1) {
  if ((!n || n === nm) && (n = kt), Uo(n)) {
    const a = Mi(
      n,
      e,
      !0
      /* mergeRef: true */
    );
    return t && No(a, t), Mr > 0 && !r && sn && (a.shapeFlag & 6 ? sn[sn.indexOf(n)] = a : sn.push(a)), a.patchFlag = -2, a;
  }
  if (Vm(n) && (n = n.__vccOpts), e) {
    e = Dm(e);
    let { class: a, style: l } = e;
    a && !xt(a) && (e.class = pr(a)), st(l) && (/* @__PURE__ */ fc(l) && !ze(l) && (l = Rt({}, l)), e.style = _i(l));
  }
  const o = xt(n) ? 1 : Hf(n) ? 128 : Ko(n) ? 64 : st(n) ? 4 : Xe(n) ? 2 : 0;
  return Be(
    n,
    e,
    t,
    i,
    s,
    o,
    r,
    !0
  );
}
function Dm(n) {
  return n ? /* @__PURE__ */ fc(n) || Df(n) ? Rt({}, n) : n : null;
}
function Mi(n, e, t = !1, i = !1) {
  const { props: s, ref: r, patchFlag: o, children: a, transition: l } = n, c = e ? Im(s || {}, e) : s, u = {
    __v_isVNode: !0,
    __v_skip: !0,
    type: n.type,
    props: c,
    key: c && Gf(c),
    ref: e && e.ref ? (
      // #2078 in the case of <component :is="vnode" ref="extra"/>
      // if the vnode itself already has a ref, cloneVNode will need to merge
      // the refs so the single vnode can be set on multiple refs
      t && r ? ze(r) ? r.concat(xo(e)) : [r, xo(e)] : xo(e)
    ) : r,
    scopeId: n.scopeId,
    slotScopeIds: n.slotScopeIds,
    children: a,
    target: n.target,
    targetStart: n.targetStart,
    targetAnchor: n.targetAnchor,
    staticCount: n.staticCount,
    shapeFlag: n.shapeFlag,
    // if the vnode is cloned with extra props, we can no longer assume its
    // existing patch flag to be reliable and need to add the FULL_PROPS flag.
    // note: preserve flag for fragments since they use the flag for children
    // fast paths only.
    patchFlag: e && n.type !== Vt ? o === -1 ? 16 : o | 16 : o,
    dynamicProps: n.dynamicProps,
    dynamicChildren: n.dynamicChildren,
    appContext: n.appContext,
    dirs: n.dirs,
    transition: l,
    // These should technically only be non-null on mounted VNodes. However,
    // they *should* be copied for kept-alive vnodes. So we just always copy
    // them since them being non-null during a mount doesn't affect the logic as
    // they will simply be overwritten.
    component: n.component,
    suspense: n.suspense,
    ssContent: n.ssContent && Mi(n.ssContent),
    ssFallback: n.ssFallback && Mi(n.ssFallback),
    placeholder: n.placeholder,
    el: n.el,
    anchor: n.anchor,
    ctx: n.ctx,
    ce: n.ce
  };
  return l && i && xr(
    u,
    l.clone(u)
  ), u;
}
function Lm(n = " ", e = 0) {
  return Kt(Qo, null, n, e);
}
function sr(n = "", e = !1) {
  return e ? (Lt(), Cm(kt, null, n)) : Kt(kt, null, n);
}
function Pn(n) {
  return n == null || typeof n == "boolean" ? Kt(kt) : ze(n) ? Kt(
    Vt,
    null,
    // #3666, avoid reference pollution when reusing vnode
    n.slice()
  ) : Uo(n) ? $n(n) : Kt(Qo, null, String(n));
}
function $n(n) {
  return n.el === null && n.patchFlag !== -1 || n.memo ? n : Mi(n);
}
function No(n, e) {
  let t = 0;
  const { shapeFlag: i } = n;
  if (e == null)
    e = null;
  else if (ze(e))
    t = 16;
  else if (typeof e == "object")
    if (i & 65) {
      const s = e.default;
      s && (s._c && (s._d = !1), No(n, s()), s._c && (s._d = !0));
      return;
    } else {
      t = 32;
      const s = e._;
      !s && !Df(e) ? e._ctx = dn : s === 3 && dn && (dn.slots._ === 1 ? e._ = 1 : (e._ = 2, n.patchFlag |= 1024));
    }
  else if (Xe(e)) {
    if (i & 65) {
      No(n, { default: e });
      return;
    }
    e = { default: e, _ctx: dn }, t = 32;
  } else
    e = String(e), i & 64 ? (t = 16, e = [Lm(e)]) : t = 8;
  n.children = e, n.shapeFlag |= t;
}
function Im(...n) {
  const e = {};
  for (let t = 0; t < n.length; t++) {
    const i = n[t];
    for (const s in i)
      if (s === "class")
        e.class !== i.class && (e.class = pr([e.class, i.class]));
      else if (s === "style")
        e.style = _i([e.style, i.style]);
      else if (Go(s)) {
        const r = e[s], o = i[s];
        o && r !== o && !(ze(r) && r.includes(o)) ? e[s] = r ? [].concat(r, o) : o : o == null && r == null && // mergeProps({ 'onUpdate:modelValue': undefined }) should not retain
        // the model listener.
        !Wo(s) && (e[s] = o);
      } else s !== "" && (e[s] = i[s]);
  }
  return e;
}
function An(n, e, t, i = null) {
  mn(n, e, 7, [
    t,
    i
  ]);
}
const Um = Af();
let Nm = 0;
function Fm(n, e, t) {
  const i = n.type, s = (e ? e.appContext : n.appContext) || Um, r = {
    uid: Nm++,
    vnode: n,
    type: i,
    parent: e,
    appContext: s,
    root: null,
    // to be immediately set
    next: null,
    subTree: null,
    // will be set synchronously right after creation
    effect: null,
    update: null,
    // will be set synchronously right after creation
    job: null,
    scope: new rp(
      !0
      /* detached */
    ),
    render: null,
    proxy: null,
    exposed: null,
    exposeProxy: null,
    withProxy: null,
    provides: e ? e.provides : Object.create(s.provides),
    ids: e ? e.ids : ["", 0, 0],
    accessCache: null,
    renderCache: [],
    // local resolved assets
    components: null,
    directives: null,
    // resolved props and emits options
    propsOptions: If(i, s),
    emitsOptions: wf(i, s),
    // emit
    emit: null,
    // to be set immediately
    emitted: null,
    // props default value
    propsDefaults: ut,
    // inheritAttrs
    inheritAttrs: i.inheritAttrs,
    // state
    ctx: ut,
    data: ut,
    props: ut,
    attrs: ut,
    slots: ut,
    refs: ut,
    setupState: ut,
    setupContext: null,
    // suspense related
    suspense: t,
    suspenseId: t ? t.pendingId : 0,
    asyncDep: null,
    asyncResolved: !1,
    // lifecycle hooks
    // not using enums here because it results in computed properties
    isMounted: !1,
    isUnmounted: !1,
    isDeactivated: !1,
    bc: null,
    c: null,
    bm: null,
    m: null,
    bu: null,
    u: null,
    um: null,
    bum: null,
    da: null,
    a: null,
    rtg: null,
    rtc: null,
    ec: null,
    sp: null
  };
  return r.ctx = { _: r }, r.root = e ? e.root : r, r.emit = fm.bind(null, r), n.ce && n.ce(r), r;
}
let Gt = null;
const Wf = () => Gt || dn;
let Fo, Sr;
{
  const n = Yo(), e = (t, i) => {
    let s;
    return (s = n[t]) || (s = n[t] = []), s.push(i), (r) => {
      s.length > 1 ? s.forEach((o) => o(r)) : s[0](r);
    };
  };
  Fo = e(
    "__VUE_INSTANCE_SETTERS__",
    (t) => Gt = t
  ), Sr = e(
    "__VUE_SSR_SETTERS__",
    (t) => yr = t
  );
}
const Dr = (n) => {
  const e = Gt;
  return Fo(n), n.scope.on(), () => {
    n.scope.off(), Fo(e);
  };
}, ru = () => {
  Gt && Gt.scope.off(), Fo(null);
};
function Xf(n) {
  return n.vnode.shapeFlag & 4;
}
let yr = !1;
function Om(n, e = !1, t = !1) {
  e && Sr(e);
  const { props: i, children: s } = n.vnode, r = Xf(n);
  vm(n, i, r, e), ym(n, s, t || e);
  const o = r ? Bm(n, e) : void 0;
  return e && Sr(!1), o;
}
function Bm(n, e) {
  const t = n.type;
  n.accessCache = /* @__PURE__ */ Object.create(null), n.proxy = new Proxy(n.ctx, im);
  const { setup: i } = t;
  if (i) {
    ni();
    const s = n.setupContext = i.length > 1 ? Hm(n) : null, r = Dr(n), o = Pr(
      i,
      n,
      0,
      [
        n.props,
        s
      ]
    ), a = Bh(o);
    if (ii(), r(), (a || n.sp) && !hr(n) && Sf(n), a) {
      if (o.then(ru, ru), e)
        return o.then((l) => {
          Sr(!0);
          try {
            ou(n, l, e);
          } finally {
            Sr(!1);
          }
        }).catch((l) => {
          jo(l, n, 0);
        });
      n.asyncDep = o;
    } else
      ou(n, o);
  } else
    Yf(n);
}
function ou(n, e, t) {
  Xe(e) ? n.type.__ssrInlineRender ? n.ssrRender = e : n.render = e : st(e) && (n.setupState = af(e)), Yf(n);
}
function Yf(n, e, t) {
  const i = n.type;
  n.render || (n.render = i.render || Fn);
  {
    const s = Dr(n);
    ni();
    try {
      sm(n);
    } finally {
      ii(), s();
    }
  }
}
const zm = {
  get(n, e) {
    return It(n, "get", ""), n[e];
  }
};
function Hm(n) {
  const e = (t) => {
    n.exposed = t || {};
  };
  return {
    attrs: new Proxy(n.attrs, zm),
    slots: n.slots,
    emit: n.emit,
    expose: e
  };
}
function ea(n) {
  return n.exposed ? n.exposeProxy || (n.exposeProxy = new Proxy(af(bp(n.exposed)), {
    get(e, t) {
      if (t in e)
        return e[t];
      if (t in fr)
        return fr[t](n);
    },
    has(e, t) {
      return t in e || t in fr;
    }
  })) : n.proxy;
}
function Vm(n) {
  return Xe(n) && "__vccOpts" in n;
}
const un = (n, e) => /* @__PURE__ */ Pp(n, e, yr);
function km(n, e, t) {
  try {
    Io(-1);
    const i = arguments.length;
    return i === 2 ? st(e) && !ze(e) ? Uo(e) ? Kt(n, null, [e]) : Kt(n, e) : Kt(n, null, e) : (i > 3 ? t = Array.prototype.slice.call(arguments, 2) : i === 3 && Uo(t) && (t = [t]), Kt(n, e, t));
  } finally {
    Io(1);
  }
}
const Gm = "3.5.41";
let hl;
const au = typeof window < "u" && window.trustedTypes;
if (au)
  try {
    hl = /* @__PURE__ */ au.createPolicy("vue", {
      createHTML: (n) => n
    });
  } catch {
  }
const qf = hl ? (n) => hl.createHTML(n) : (n) => n, Wm = "http://www.w3.org/2000/svg", Xm = "http://www.w3.org/1998/Math/MathML", Kn = typeof document < "u" ? document : null, lu = Kn && /* @__PURE__ */ Kn.createElement("template"), Ym = {
  insert: (n, e, t) => {
    e.insertBefore(n, t || null);
  },
  remove: (n) => {
    const e = n.parentNode;
    e && e.removeChild(n);
  },
  createElement: (n, e, t, i) => {
    const s = e === "svg" ? Kn.createElementNS(Wm, n) : e === "mathml" ? Kn.createElementNS(Xm, n) : t ? Kn.createElement(n, { is: t }) : Kn.createElement(n);
    return n === "select" && i && i.multiple != null && s.setAttribute("multiple", i.multiple), s;
  },
  createText: (n) => Kn.createTextNode(n),
  createComment: (n) => Kn.createComment(n),
  setText: (n, e) => {
    n.nodeValue = e;
  },
  setElementText: (n, e) => {
    n.textContent = e;
  },
  parentNode: (n) => n.parentNode,
  nextSibling: (n) => n.nextSibling,
  querySelector: (n) => Kn.querySelector(n),
  setScopeId(n, e) {
    n.setAttribute(e, "");
  },
  // __UNSAFE__
  // Reason: innerHTML.
  // Static content here can only come from compiled templates.
  // As long as the user only uses trusted templates, this is safe.
  insertStaticContent(n, e, t, i, s, r) {
    const o = t ? t.previousSibling : e.lastChild;
    if (s && (s === r || s.nextSibling))
      for (; e.insertBefore(s.cloneNode(!0), t), !(s === r || !(s = s.nextSibling)); )
        ;
    else {
      lu.innerHTML = qf(
        i === "svg" ? `<svg>${n}</svg>` : i === "mathml" ? `<math>${n}</math>` : n
      );
      const a = lu.content;
      if (i === "svg" || i === "mathml") {
        const l = a.firstChild;
        for (; l.firstChild; )
          a.appendChild(l.firstChild);
        a.removeChild(l);
      }
      e.insertBefore(a, t);
    }
    return [
      // first
      o ? o.nextSibling : e.firstChild,
      // last
      t ? t.previousSibling : e.lastChild
    ];
  }
}, oi = "transition", qs = "animation", Er = /* @__PURE__ */ Symbol("_vtc"), jf = {
  name: String,
  type: String,
  css: {
    type: Boolean,
    default: !0
  },
  duration: [String, Number, Object],
  enterFromClass: String,
  enterActiveClass: String,
  enterToClass: String,
  appearFromClass: String,
  appearActiveClass: String,
  appearToClass: String,
  leaveFromClass: String,
  leaveActiveClass: String,
  leaveToClass: String
}, qm = /* @__PURE__ */ Rt(
  {},
  _f,
  jf
), jm = (n) => (n.displayName = "Transition", n.props = qm, n), Km = /* @__PURE__ */ jm(
  (n, { slots: e }) => km(Xp, $m(n), e)
), wi = (n, e = []) => {
  ze(n) ? n.forEach((t) => t(...e)) : n && n(...e);
}, cu = (n) => n ? ze(n) ? n.some((e) => e.length > 1) : n.length > 1 : !1;
function $m(n) {
  const e = {};
  for (const L in n)
    L in jf || (e[L] = n[L]);
  if (n.css === !1)
    return e;
  const {
    name: t = "v",
    type: i,
    duration: s,
    enterFromClass: r = `${t}-enter-from`,
    enterActiveClass: o = `${t}-enter-active`,
    enterToClass: a = `${t}-enter-to`,
    appearFromClass: l = r,
    appearActiveClass: c = o,
    appearToClass: u = a,
    leaveFromClass: h = `${t}-leave-from`,
    leaveActiveClass: f = `${t}-leave-active`,
    leaveToClass: p = `${t}-leave-to`
  } = n, v = Zm(s), x = v && v[0], m = v && v[1], {
    onBeforeEnter: d,
    onEnter: b,
    onEnterCancelled: A,
    onLeave: M,
    onLeaveCancelled: R,
    onBeforeAppear: w = d,
    onAppear: D = b,
    onAppearCancelled: U = A
  } = e, y = (L, V, Z, te) => {
    L._enterCancelled = te, Ri(L, V ? u : a), Ri(L, V ? c : o), Z && Z();
  }, S = (L, V) => {
    L._isLeaving = !1, Ri(L, h), Ri(L, p), Ri(L, f), V && V();
  }, P = (L) => (V, Z) => {
    const te = L ? D : b, $ = () => y(V, L, Z);
    wi(te, [V, $]), uu(() => {
      Ri(V, L ? l : r), kn(V, L ? u : a), cu(te) || hu(V, i, x, $);
    });
  };
  return Rt(e, {
    onBeforeEnter(L) {
      wi(d, [L]), kn(L, r), kn(L, o);
    },
    onBeforeAppear(L) {
      wi(w, [L]), kn(L, l), kn(L, c);
    },
    onEnter: P(!1),
    onAppear: P(!0),
    onLeave(L, V) {
      L._isLeaving = !0;
      const Z = () => S(L, V);
      kn(L, h), L._enterCancelled ? (kn(L, f), pu(L)) : (pu(L), kn(L, f)), uu(() => {
        L._isLeaving && (Ri(L, h), kn(L, p), cu(M) || hu(L, i, m, Z));
      }), wi(M, [L, Z]);
    },
    onEnterCancelled(L) {
      y(L, !1, void 0, !0), wi(A, [L]);
    },
    onAppearCancelled(L) {
      y(L, !0, void 0, !0), wi(U, [L]);
    },
    onLeaveCancelled(L) {
      S(L), wi(R, [L]);
    }
  });
}
function Zm(n) {
  if (n == null)
    return null;
  if (st(n))
    return [_a(n.enter), _a(n.leave)];
  {
    const e = _a(n);
    return [e, e];
  }
}
function _a(n) {
  return Zd(n);
}
function kn(n, e) {
  e.split(/\s+/).forEach((t) => t && n.classList.add(t)), (n[Er] || (n[Er] = /* @__PURE__ */ new Set())).add(e);
}
function Ri(n, e) {
  e.split(/\s+/).forEach((i) => i && n.classList.remove(i));
  const t = n[Er];
  t && (t.delete(e), t.size || (n[Er] = void 0));
}
function uu(n) {
  requestAnimationFrame(() => {
    requestAnimationFrame(n);
  });
}
let Jm = 0;
function hu(n, e, t, i) {
  const s = n._endId = ++Jm, r = () => {
    s === n._endId && i();
  };
  if (t != null)
    return setTimeout(r, t);
  const { type: o, timeout: a, propCount: l } = Qm(n, e);
  if (!o)
    return i();
  const c = o + "end";
  let u = 0;
  const h = () => {
    n.removeEventListener(c, f), r();
  }, f = (p) => {
    p.target === n && ++u >= l && h();
  };
  setTimeout(() => {
    u < l && h();
  }, a + 1), n.addEventListener(c, f);
}
function Qm(n, e) {
  const t = window.getComputedStyle(n), i = (v) => (t[v] || "").split(", "), s = i(`${oi}Delay`), r = i(`${oi}Duration`), o = fu(s, r), a = i(`${qs}Delay`), l = i(`${qs}Duration`), c = fu(a, l);
  let u = null, h = 0, f = 0;
  e === oi ? o > 0 && (u = oi, h = o, f = r.length) : e === qs ? c > 0 && (u = qs, h = c, f = l.length) : (h = Math.max(o, c), u = h > 0 ? o > c ? oi : qs : null, f = u ? u === oi ? r.length : l.length : 0);
  const p = u === oi && /\b(?:transform|all)(?:,|$)/.test(
    i(`${oi}Property`).toString()
  );
  return {
    type: u,
    timeout: h,
    propCount: f,
    hasTransform: p
  };
}
function fu(n, e) {
  for (; n.length < e.length; )
    n = n.concat(n);
  return Math.max(...e.map((t, i) => du(t) + du(n[i])));
}
function du(n) {
  return n === "auto" ? 0 : Number(n.slice(0, -1).replace(",", ".")) * 1e3;
}
function pu(n) {
  return (n ? n.ownerDocument : document).body.offsetHeight;
}
function e_(n, e, t) {
  const i = n[Er];
  i && (e = (e ? [e, ...i] : [...i]).join(" ")), e == null ? n.removeAttribute("class") : t ? n.setAttribute("class", e) : n.className = e;
}
const Oo = /* @__PURE__ */ Symbol("_vod"), Kf = /* @__PURE__ */ Symbol("_vsh"), t_ = {
  // used for prop mismatch check during hydration
  name: "show",
  beforeMount(n, { value: e }, { transition: t }) {
    n[Oo] = n.style.display === "none" ? "" : n.style.display, t && e ? t.beforeEnter(n) : js(n, e);
  },
  mounted(n, { value: e }, { transition: t }) {
    t && e && t.enter(n);
  },
  updated(n, { value: e, oldValue: t }, { transition: i }) {
    !e != !t && (i ? e ? (i.beforeEnter(n), js(n, !0), i.enter(n)) : i.leave(n, () => {
      js(n, !1);
    }) : js(n, e));
  },
  beforeUnmount(n, { value: e }) {
    js(n, e);
  }
};
function js(n, e) {
  n.style.display = e ? n[Oo] : "none", n[Kf] = !e;
}
const n_ = /* @__PURE__ */ Symbol(""), i_ = /(?:^|;)\s*display\s*:/;
function s_(n, e, t) {
  const i = n.style, s = xt(t);
  let r = !1;
  if (t && !s) {
    if (e)
      if (xt(e))
        for (const o of e.split(";")) {
          const a = o.slice(0, o.indexOf(":")).trim();
          t[a] == null && rr(i, a, "");
        }
      else
        for (const o in e)
          t[o] == null && rr(i, o, "");
    for (const o in t) {
      o === "display" && (r = !0);
      const a = t[o];
      a != null ? o_(
        n,
        o,
        !xt(e) && e ? e[o] : void 0,
        a
      ) || rr(i, o, a) : rr(i, o, "");
    }
  } else if (s) {
    if (e !== t) {
      const o = i[n_];
      o && (t += ";" + o), i.cssText = t, r = i_.test(t);
    }
  } else e && n.removeAttribute("style");
  Oo in n && (n[Oo] = r ? i.display : "", n[Kf] && (i.display = "none"));
}
const mu = /\s*!important$/;
function rr(n, e, t) {
  if (ze(t))
    t.forEach((i) => rr(n, e, i));
  else if (t == null && (t = ""), e.startsWith("--"))
    n.setProperty(e, t);
  else {
    const i = r_(n, e);
    mu.test(t) ? n.setProperty(
      Ki(i),
      t.replace(mu, ""),
      "important"
    ) : n[i] = t;
  }
}
const _u = ["Webkit", "Moz", "ms"], ga = {};
function r_(n, e) {
  const t = ga[e];
  if (t)
    return t;
  let i = Mn(e);
  if (i !== "filter" && i in n)
    return ga[e] = i;
  i = Vh(i);
  for (let s = 0; s < _u.length; s++) {
    const r = _u[s] + i;
    if (r in n)
      return ga[e] = r;
  }
  return e;
}
function o_(n, e, t, i) {
  return n.tagName === "TEXTAREA" && (e === "width" || e === "height") && xt(i) && t === i;
}
const gu = "http://www.w3.org/1999/xlink";
function vu(n, e, t, i, s, r = ip(e)) {
  i && e.startsWith("xlink:") ? t == null ? n.removeAttributeNS(gu, e.slice(6, e.length)) : n.setAttributeNS(gu, e, t) : t == null || r && !Gh(t) ? n.removeAttribute(e) : n.setAttribute(
    e,
    r ? "" : On(t) ? String(t) : t
  );
}
function xu(n, e, t, i, s) {
  if (e === "innerHTML" || e === "textContent") {
    t != null && (n[e] = e === "innerHTML" ? qf(t) : t);
    return;
  }
  const r = n.tagName;
  if (e === "value" && r !== "PROGRESS" && // custom elements may use _value internally
  !r.includes("-")) {
    const a = r === "OPTION" ? n.getAttribute("value") || "" : n.value, l = t == null ? (
      // #11647: value should be set as empty string for null and undefined,
      // but <input type="checkbox"> should be set as 'on'.
      n.type === "checkbox" ? "on" : ""
    ) : String(t);
    (a !== l || !("_value" in n)) && (n.value = l), t == null && n.removeAttribute(e), n._value = t;
    return;
  }
  let o = !1;
  if (t === "" || t == null) {
    const a = typeof n[e];
    a === "boolean" ? t = Gh(t) : t == null && a === "string" ? (t = "", o = !0) : a === "number" && (t = 0, o = !0);
  }
  try {
    n[e] = t;
  } catch {
  }
  o && n.removeAttribute(s || e);
}
function a_(n, e, t, i) {
  n.addEventListener(e, t, i);
}
function l_(n, e, t, i) {
  n.removeEventListener(e, t, i);
}
const Mu = /* @__PURE__ */ Symbol("_vei");
function c_(n, e, t, i, s = null) {
  const r = n[Mu] || (n[Mu] = {}), o = r[e];
  if (i && o)
    o.value = i;
  else {
    const [a, l] = f_(e);
    if (i) {
      const c = r[e] = m_(
        i,
        s
      );
      a_(n, a, c, l);
    } else o && (l_(n, a, o, l), r[e] = void 0);
  }
}
const u_ = /(Once|Passive|Capture)$/, h_ = /^on:?(?:Once|Passive|Capture)$/;
function f_(n) {
  let e, t;
  for (; (t = n.match(u_)) && !h_.test(n); )
    e || (e = {}), n = n.slice(0, n.length - t[1].length), e[t[1].toLowerCase()] = !0;
  return [n[2] === ":" ? n.slice(3) : Ki(n.slice(2)), e];
}
let va = 0;
const d_ = /* @__PURE__ */ Promise.resolve(), p_ = () => va || (d_.then(() => va = 0), va = Date.now());
function m_(n, e) {
  const t = (i) => {
    if (!i._vts)
      i._vts = Date.now();
    else if (i._vts <= t.attached)
      return;
    const s = t.value;
    if (ze(s)) {
      const r = i.stopImmediatePropagation;
      i.stopImmediatePropagation = () => {
        r.call(i), i._stopped = !0;
      };
      const o = s.slice(), a = [i];
      for (let l = 0; l < o.length && !i._stopped; l++) {
        const c = o[l];
        c && mn(
          c,
          e,
          5,
          a
        );
      }
    } else
      mn(
        s,
        e,
        5,
        [i]
      );
  };
  return t.value = n, t.attached = p_(), t;
}
const Su = (n) => n.charCodeAt(0) === 111 && n.charCodeAt(1) === 110 && // lowercase letter
n.charCodeAt(2) > 96 && n.charCodeAt(2) < 123, __ = (n, e, t, i, s, r) => {
  const o = s === "svg";
  e === "class" ? e_(n, i, o) : e === "style" ? s_(n, t, i) : Go(e) ? Wo(e) || c_(n, e, t, i, r) : (e[0] === "." ? (e = e.slice(1), !0) : e[0] === "^" ? (e = e.slice(1), !1) : g_(n, e, i, o)) ? (xu(n, e, i), !n.tagName.includes("-") && (e === "value" || e === "checked" || e === "selected") && vu(n, e, i, o, r, e !== "value")) : /* #11081 force set props for possible async custom element */ n._isVueCE && // #12408 check if it's declared prop or it's async custom element
  (v_(n, e) || // @ts-expect-error _def is private
  n._def.__asyncLoader && (/[A-Z]/.test(e) || !xt(i))) ? xu(n, Mn(e), i, r, e) : (e === "true-value" ? n._trueValue = i : e === "false-value" && (n._falseValue = i), vu(n, e, i, o));
};
function g_(n, e, t, i) {
  if (i)
    return !!(e === "innerHTML" || e === "textContent" || e in n && Su(e) && Xe(t));
  if (e === "spellcheck" || e === "draggable" || e === "translate" || e === "autocorrect" || e === "sandbox" && n.tagName === "IFRAME" || e === "form" || e === "list" && n.tagName === "INPUT" || e === "type" && n.tagName === "TEXTAREA")
    return !1;
  if (e === "width" || e === "height") {
    const s = n.tagName;
    if (s === "IMG" || s === "VIDEO" || s === "CANVAS" || s === "SOURCE")
      return !1;
  }
  return Su(e) && xt(t) ? !1 : e in n;
}
function v_(n, e) {
  const t = (
    // @ts-expect-error _def is private
    n._def.props
  );
  if (!t)
    return !1;
  const i = Mn(e);
  return Array.isArray(t) ? t.some((s) => Mn(s) === i) : Object.keys(t).some((s) => Mn(s) === i);
}
const x_ = ["ctrl", "shift", "alt", "meta"], M_ = {
  stop: (n) => n.stopPropagation(),
  prevent: (n) => n.preventDefault(),
  self: (n) => n.target !== n.currentTarget,
  ctrl: (n) => !n.ctrlKey,
  shift: (n) => !n.shiftKey,
  alt: (n) => !n.altKey,
  meta: (n) => !n.metaKey,
  left: (n) => "button" in n && n.button !== 0,
  middle: (n) => "button" in n && n.button !== 1,
  right: (n) => "button" in n && n.button !== 2,
  exact: (n, e) => x_.some((t) => n[`${t}Key`] && !e.includes(t))
}, S_ = (n, e) => {
  if (!n) return n;
  const t = n._withMods || (n._withMods = {}), i = e.join(".");
  return t[i] || (t[i] = ((s, ...r) => {
    for (let o = 0; o < e.length; o++) {
      const a = M_[e[o]];
      if (a && a(s, e)) return;
    }
    return n(s, ...r);
  }));
}, y_ = /* @__PURE__ */ Rt({ patchProp: __ }, Ym);
let yu;
function E_() {
  return yu || (yu = Tm(y_));
}
const T_ = ((...n) => {
  const e = E_().createApp(...n), { mount: t } = e;
  return e.mount = (i) => {
    const s = A_(i);
    if (!s) return;
    const r = e._component;
    !Xe(r) && !r.render && !r.template && (r.template = s.innerHTML), s.nodeType === 1 && (s.textContent = "");
    const o = t(s, !1, b_(s));
    return s instanceof Element && (s.removeAttribute("v-cloak"), s.setAttribute("data-v-app", "")), o;
  }, e;
});
function b_(n) {
  if (n instanceof SVGElement)
    return "svg";
  if (typeof MathMLElement == "function" && n instanceof MathMLElement)
    return "mathml";
}
function A_(n) {
  return xt(n) ? document.querySelector(n) : n;
}
const w_ = "AKUSPACE", R_ = "ltx25_audio", C_ = [{ id: "ltx25_audio", label: "LTX-2.5 Audio", trigger: "AKUSPACE", status: "active", supported_modes: ["dry", "room", "outside", "sfx"] }], P_ = { modes: [{ value: "dry", label: "Off" }, { value: "room", label: "Room" }, { value: "outside", label: "Space" }, { value: "sfx", label: "Sound effects" }], room_presets: ["small_room", "empty_club", "medium_room", "cathedral"], reverb_levels: ["low", "mid", "high"], outdoor_times: ["day", "night"], outdoor_level: "low", sfx_presets: ["dual_delay"], sfx_levels: ["low", "high"] }, D_ = { low: { label: "Low", caption_word: "gentle", relative_db: -25, visual_amount: 0.28 }, mid: { label: "Moderate", caption_word: "moderate", relative_db: -12, visual_amount: 0.58 }, high: { label: "Heavy", caption_word: "heavy", relative_db: 0, visual_amount: 1 } }, L_ = { dry: { mode: "dry", label: "Dry / off", short_label: "Dry", description: "Bypass the LoRA and keep the reference dry.", acoustic_fingerprint: "dry reference · no reverb", dimensions_m: [4, 5, 2.8], estimated_rt60: 0.08, estimated_predelay_ms: 1, palette: ["#d7ded9", "#202522"] }, small_room: { mode: "room", label: "Small room", short_label: "Small", description: "Bathroom-scale space with bright, close reflections.", acoustic_fingerprint: "trained caption 0.67 s · source setting ≈1.07 s", dimensions_m: [2.4, 3.2, 2.5], caption_where: "in a small bathroom-like room", caption_character: "bright close reflections and a short 0.67-second reverb decay", caption_tail: "no background ambience", estimated_rt60: 0.67, estimated_predelay_ms: 4, palette: ["#8edcff", "#173344"] }, medium_room: { mode: "room", label: "Medium room", short_label: "Medium", description: "Balanced room scale with a smooth 1.9-second decay.", acoustic_fingerprint: "trained caption 1.90 s · source setting ≈1.92 s", dimensions_m: [7, 9, 3.6], caption_where: "in a medium reverberant room", caption_character: "smooth reflections and a 1.9-second reverb decay", caption_tail: "no background ambience", estimated_rt60: 1.9, estimated_predelay_ms: 11, palette: ["#82f5c2", "#15372e"] }, empty_club: { mode: "room", label: "Empty club", short_label: "Club", description: "Filtered hard-surface reflections with a shorter, tighter 1.2-second decay.", acoustic_fingerprint: "1.20 s decay · device size 6.8", dimensions_m: [16, 24, 6], caption_where: "in an empty club", caption_character: "broad hard-surface reflections and a 1.2-second reverb decay", caption_tail: "no crowd", estimated_rt60: 1.2, estimated_predelay_ms: 18, palette: ["#ffb86f", "#422918"] }, cathedral: { mode: "room", label: "Cathedral", short_label: "Cathedral", description: "Monumental synthetic space with a long diffuse tail.", acoustic_fingerprint: "synthetic cathedral · 506 ms delay setting", dimensions_m: [26, 78, 29], caption_where: "through synthetic cathedral reverb", caption_character: "wide diffuse reflections and a long decaying tail", caption_tail: "no background ambience", estimated_rt60: 4.8, estimated_predelay_ms: 38, palette: ["#d6adff", "#392c4e"] }, dual_delay: { mode: "sfx", effect_type: "modular_dual_delay", coverage: "experimental", label: "Dual Delay", short_label: "Dual Delay", description: "Experimental modular dual-delay patch captioned during training as a modular granular delay.", acoustic_fingerprint: "modular dual delay · modular granular training caption", dimensions_m: [10, 16, 4], caption_where: "through a modular granular delay", caption_character: "scattered grains and unpredictable modulated echoes", caption_tail: "no background ambience", estimated_rt60: 0, estimated_predelay_ms: 0, palette: ["#ff78e8", "#43183e"] }, outdoor_day: { mode: "outside", time_of_day: "day", label: "Outside · day", short_label: "Day", description: "Open-air acoustics with a continuous birdsong bed.", acoustic_fingerprint: "day birds · fixed trained ambience", dimensions_m: [60, 120, 30], caption_where: "outdoors in daytime", caption_character: "open-air acoustics with continuous birdsong ambience", estimated_rt60: 0.08, estimated_predelay_ms: 2, palette: ["#ffc95c", "#3d331d"] }, outdoor_night: { mode: "outside", time_of_day: "night", label: "Outside · night", short_label: "Night", description: "Open-air acoustics with crickets and distant cars.", acoustic_fingerprint: "night crickets + cars · fixed trained ambience", dimensions_m: [60, 120, 30], caption_where: "outdoors at night", caption_character: "open-air acoustics with crickets and distant car ambience", estimated_rt60: 0.1, estimated_predelay_ms: 2, palette: ["#8f9cff", "#141a3b"] } }, Hs = {
  trigger: w_,
  default_model_profile: R_,
  model_profiles: C_,
  control_schema: P_,
  levels: D_,
  presets: L_
}, $f = Hs.presets, Mo = Hs.levels, $i = Hs.control_schema, Eu = Hs.model_profiles, I_ = Hs.default_model_profile, U_ = Eu.find(
  (n) => n.id === I_
) ?? Eu[0];
U_?.trigger ?? Hs.trigger;
const ts = $i.modes, vs = $i.room_presets, xs = $i.reverb_levels, N_ = $i.outdoor_times, F_ = $i.outdoor_level, O_ = $i.sfx_presets, Ms = $i.sfx_levels, pi = {
  space_mode: "room",
  room_preset: "medium_room",
  outdoor_time: "day",
  sfx_preset: "dual_delay",
  effect_level: "mid",
  sfx_level: "low",
  source_type: "male spoken voice"
};
function B_(n) {
  return n.space_mode === "dry" ? "dry" : n.space_mode === "outside" ? n.outdoor_time === "night" ? "outdoor_night" : "outdoor_day" : n.space_mode === "sfx" ? O_.includes(n.sfx_preset) ? n.sfx_preset : "dual_delay" : vs.includes(n.room_preset) ? n.room_preset : "medium_room";
}
function Zf(n) {
  return $f[B_(n)];
}
function z_(n) {
  return n.space_mode === "outside" ? F_ : n.space_mode === "sfx" ? Ms.includes(n.sfx_level) ? n.sfx_level : Ms[0] : xs.includes(n.effect_level) ? n.effect_level : "mid";
}
function H_(n) {
  const e = Zf(n), t = Mo[z_(n)], [i, s, r] = e.dimensions_m;
  return {
    rt60: e.estimated_rt60,
    predelay_ms: e.estimated_predelay_ms,
    volume_m3: i * s * r,
    visual_amount: n.space_mode === "dry" ? 0 : t.visual_amount
  };
}
const vc = "180", Ps = { ROTATE: 0, DOLLY: 1, PAN: 2 }, Ss = { ROTATE: 0, PAN: 1, DOLLY_PAN: 2, DOLLY_ROTATE: 3 }, V_ = 0, Tu = 1, k_ = 2, Jf = 1, G_ = 2, jn = 3, Si = 0, Wt = 1, Qn = 2, vi = 0, Ds = 1, bu = 2, Au = 3, wu = 4, W_ = 5, Bi = 100, X_ = 101, Y_ = 102, q_ = 103, j_ = 104, K_ = 200, $_ = 201, Z_ = 202, J_ = 203, fl = 204, dl = 205, Q_ = 206, eg = 207, tg = 208, ng = 209, ig = 210, sg = 211, rg = 212, og = 213, ag = 214, pl = 0, ml = 1, _l = 2, Us = 3, gl = 4, vl = 5, xl = 6, Ml = 7, Qf = 0, lg = 1, cg = 2, xi = 0, ug = 1, hg = 2, fg = 3, ed = 4, dg = 5, pg = 6, mg = 7, td = 300, Ns = 301, Fs = 302, Sl = 303, yl = 304, ta = 306, El = 1e3, Hi = 1001, Tl = 1002, yn = 1003, _g = 1004, Vr = 1005, Un = 1006, xa = 1007, Vi = 1008, Bn = 1009, nd = 1010, id = 1011, Tr = 1012, xc = 1013, Xi = 1014, ei = 1015, Lr = 1016, Mc = 1017, Sc = 1018, br = 1020, sd = 35902, rd = 35899, od = 1021, ad = 1022, xn = 1023, Ar = 1026, wr = 1027, ld = 1028, yc = 1029, cd = 1030, Ec = 1031, Tc = 1033, So = 33776, yo = 33777, Eo = 33778, To = 33779, bl = 35840, Al = 35841, wl = 35842, Rl = 35843, Cl = 36196, Pl = 37492, Dl = 37496, Ll = 37808, Il = 37809, Ul = 37810, Nl = 37811, Fl = 37812, Ol = 37813, Bl = 37814, zl = 37815, Hl = 37816, Vl = 37817, kl = 37818, Gl = 37819, Wl = 37820, Xl = 37821, Yl = 36492, ql = 36494, jl = 36495, Kl = 36283, $l = 36284, Zl = 36285, Jl = 36286, gg = 3200, vg = 3201, ud = 0, xg = 1, gi = "", tn = "srgb", Os = "srgb-linear", Bo = "linear", ot = "srgb", ns = 7680, Ru = 519, Mg = 512, Sg = 513, yg = 514, hd = 515, Eg = 516, Tg = 517, bg = 518, Ag = 519, Cu = 35044, Pu = "300 es", Nn = 2e3, zo = 2001;
class Zi {
  /**
   * Adds the given event listener to the given event type.
   *
   * @param {string} type - The type of event to listen to.
   * @param {Function} listener - The function that gets called when the event is fired.
   */
  addEventListener(e, t) {
    this._listeners === void 0 && (this._listeners = {});
    const i = this._listeners;
    i[e] === void 0 && (i[e] = []), i[e].indexOf(t) === -1 && i[e].push(t);
  }
  /**
   * Returns `true` if the given event listener has been added to the given event type.
   *
   * @param {string} type - The type of event.
   * @param {Function} listener - The listener to check.
   * @return {boolean} Whether the given event listener has been added to the given event type.
   */
  hasEventListener(e, t) {
    const i = this._listeners;
    return i === void 0 ? !1 : i[e] !== void 0 && i[e].indexOf(t) !== -1;
  }
  /**
   * Removes the given event listener from the given event type.
   *
   * @param {string} type - The type of event.
   * @param {Function} listener - The listener to remove.
   */
  removeEventListener(e, t) {
    const i = this._listeners;
    if (i === void 0) return;
    const s = i[e];
    if (s !== void 0) {
      const r = s.indexOf(t);
      r !== -1 && s.splice(r, 1);
    }
  }
  /**
   * Dispatches an event object.
   *
   * @param {Object} event - The event that gets fired.
   */
  dispatchEvent(e) {
    const t = this._listeners;
    if (t === void 0) return;
    const i = t[e.type];
    if (i !== void 0) {
      e.target = this;
      const s = i.slice(0);
      for (let r = 0, o = s.length; r < o; r++)
        s[r].call(this, e);
      e.target = null;
    }
  }
}
const Pt = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "0a", "0b", "0c", "0d", "0e", "0f", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "1a", "1b", "1c", "1d", "1e", "1f", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "2a", "2b", "2c", "2d", "2e", "2f", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "3a", "3b", "3c", "3d", "3e", "3f", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "4a", "4b", "4c", "4d", "4e", "4f", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5a", "5b", "5c", "5d", "5e", "5f", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "6a", "6b", "6c", "6d", "6e", "6f", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7a", "7b", "7c", "7d", "7e", "7f", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "8a", "8b", "8c", "8d", "8e", "8f", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "9a", "9b", "9c", "9d", "9e", "9f", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "aa", "ab", "ac", "ad", "ae", "af", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "ba", "bb", "bc", "bd", "be", "bf", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "ca", "cb", "cc", "cd", "ce", "cf", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "da", "db", "dc", "dd", "de", "df", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "ea", "eb", "ec", "ed", "ee", "ef", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "fa", "fb", "fc", "fd", "fe", "ff"], dr = Math.PI / 180, Ql = 180 / Math.PI;
function Ir() {
  const n = Math.random() * 4294967295 | 0, e = Math.random() * 4294967295 | 0, t = Math.random() * 4294967295 | 0, i = Math.random() * 4294967295 | 0;
  return (Pt[n & 255] + Pt[n >> 8 & 255] + Pt[n >> 16 & 255] + Pt[n >> 24 & 255] + "-" + Pt[e & 255] + Pt[e >> 8 & 255] + "-" + Pt[e >> 16 & 15 | 64] + Pt[e >> 24 & 255] + "-" + Pt[t & 63 | 128] + Pt[t >> 8 & 255] + "-" + Pt[t >> 16 & 255] + Pt[t >> 24 & 255] + Pt[i & 255] + Pt[i >> 8 & 255] + Pt[i >> 16 & 255] + Pt[i >> 24 & 255]).toLowerCase();
}
function je(n, e, t) {
  return Math.max(e, Math.min(t, n));
}
function wg(n, e) {
  return (n % e + e) % e;
}
function Ma(n, e, t) {
  return (1 - t) * n + t * e;
}
function Ks(n, e) {
  switch (e.constructor) {
    case Float32Array:
      return n;
    case Uint32Array:
      return n / 4294967295;
    case Uint16Array:
      return n / 65535;
    case Uint8Array:
      return n / 255;
    case Int32Array:
      return Math.max(n / 2147483647, -1);
    case Int16Array:
      return Math.max(n / 32767, -1);
    case Int8Array:
      return Math.max(n / 127, -1);
    default:
      throw new Error("Invalid component type.");
  }
}
function Yt(n, e) {
  switch (e.constructor) {
    case Float32Array:
      return n;
    case Uint32Array:
      return Math.round(n * 4294967295);
    case Uint16Array:
      return Math.round(n * 65535);
    case Uint8Array:
      return Math.round(n * 255);
    case Int32Array:
      return Math.round(n * 2147483647);
    case Int16Array:
      return Math.round(n * 32767);
    case Int8Array:
      return Math.round(n * 127);
    default:
      throw new Error("Invalid component type.");
  }
}
const Rg = {
  DEG2RAD: dr
};
class Ve {
  /**
   * Constructs a new 2D vector.
   *
   * @param {number} [x=0] - The x value of this vector.
   * @param {number} [y=0] - The y value of this vector.
   */
  constructor(e = 0, t = 0) {
    Ve.prototype.isVector2 = !0, this.x = e, this.y = t;
  }
  /**
   * Alias for {@link Vector2#x}.
   *
   * @type {number}
   */
  get width() {
    return this.x;
  }
  set width(e) {
    this.x = e;
  }
  /**
   * Alias for {@link Vector2#y}.
   *
   * @type {number}
   */
  get height() {
    return this.y;
  }
  set height(e) {
    this.y = e;
  }
  /**
   * Sets the vector components.
   *
   * @param {number} x - The value of the x component.
   * @param {number} y - The value of the y component.
   * @return {Vector2} A reference to this vector.
   */
  set(e, t) {
    return this.x = e, this.y = t, this;
  }
  /**
   * Sets the vector components to the same value.
   *
   * @param {number} scalar - The value to set for all vector components.
   * @return {Vector2} A reference to this vector.
   */
  setScalar(e) {
    return this.x = e, this.y = e, this;
  }
  /**
   * Sets the vector's x component to the given value
   *
   * @param {number} x - The value to set.
   * @return {Vector2} A reference to this vector.
   */
  setX(e) {
    return this.x = e, this;
  }
  /**
   * Sets the vector's y component to the given value
   *
   * @param {number} y - The value to set.
   * @return {Vector2} A reference to this vector.
   */
  setY(e) {
    return this.y = e, this;
  }
  /**
   * Allows to set a vector component with an index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y.
   * @param {number} value - The value to set.
   * @return {Vector2} A reference to this vector.
   */
  setComponent(e, t) {
    switch (e) {
      case 0:
        this.x = t;
        break;
      case 1:
        this.y = t;
        break;
      default:
        throw new Error("index is out of range: " + e);
    }
    return this;
  }
  /**
   * Returns the value of the vector component which matches the given index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y.
   * @return {number} A vector component value.
   */
  getComponent(e) {
    switch (e) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      default:
        throw new Error("index is out of range: " + e);
    }
  }
  /**
   * Returns a new vector with copied values from this instance.
   *
   * @return {Vector2} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.x, this.y);
  }
  /**
   * Copies the values of the given vector to this instance.
   *
   * @param {Vector2} v - The vector to copy.
   * @return {Vector2} A reference to this vector.
   */
  copy(e) {
    return this.x = e.x, this.y = e.y, this;
  }
  /**
   * Adds the given vector to this instance.
   *
   * @param {Vector2} v - The vector to add.
   * @return {Vector2} A reference to this vector.
   */
  add(e) {
    return this.x += e.x, this.y += e.y, this;
  }
  /**
   * Adds the given scalar value to all components of this instance.
   *
   * @param {number} s - The scalar to add.
   * @return {Vector2} A reference to this vector.
   */
  addScalar(e) {
    return this.x += e, this.y += e, this;
  }
  /**
   * Adds the given vectors and stores the result in this instance.
   *
   * @param {Vector2} a - The first vector.
   * @param {Vector2} b - The second vector.
   * @return {Vector2} A reference to this vector.
   */
  addVectors(e, t) {
    return this.x = e.x + t.x, this.y = e.y + t.y, this;
  }
  /**
   * Adds the given vector scaled by the given factor to this instance.
   *
   * @param {Vector2} v - The vector.
   * @param {number} s - The factor that scales `v`.
   * @return {Vector2} A reference to this vector.
   */
  addScaledVector(e, t) {
    return this.x += e.x * t, this.y += e.y * t, this;
  }
  /**
   * Subtracts the given vector from this instance.
   *
   * @param {Vector2} v - The vector to subtract.
   * @return {Vector2} A reference to this vector.
   */
  sub(e) {
    return this.x -= e.x, this.y -= e.y, this;
  }
  /**
   * Subtracts the given scalar value from all components of this instance.
   *
   * @param {number} s - The scalar to subtract.
   * @return {Vector2} A reference to this vector.
   */
  subScalar(e) {
    return this.x -= e, this.y -= e, this;
  }
  /**
   * Subtracts the given vectors and stores the result in this instance.
   *
   * @param {Vector2} a - The first vector.
   * @param {Vector2} b - The second vector.
   * @return {Vector2} A reference to this vector.
   */
  subVectors(e, t) {
    return this.x = e.x - t.x, this.y = e.y - t.y, this;
  }
  /**
   * Multiplies the given vector with this instance.
   *
   * @param {Vector2} v - The vector to multiply.
   * @return {Vector2} A reference to this vector.
   */
  multiply(e) {
    return this.x *= e.x, this.y *= e.y, this;
  }
  /**
   * Multiplies the given scalar value with all components of this instance.
   *
   * @param {number} scalar - The scalar to multiply.
   * @return {Vector2} A reference to this vector.
   */
  multiplyScalar(e) {
    return this.x *= e, this.y *= e, this;
  }
  /**
   * Divides this instance by the given vector.
   *
   * @param {Vector2} v - The vector to divide.
   * @return {Vector2} A reference to this vector.
   */
  divide(e) {
    return this.x /= e.x, this.y /= e.y, this;
  }
  /**
   * Divides this vector by the given scalar.
   *
   * @param {number} scalar - The scalar to divide.
   * @return {Vector2} A reference to this vector.
   */
  divideScalar(e) {
    return this.multiplyScalar(1 / e);
  }
  /**
   * Multiplies this vector (with an implicit 1 as the 3rd component) by
   * the given 3x3 matrix.
   *
   * @param {Matrix3} m - The matrix to apply.
   * @return {Vector2} A reference to this vector.
   */
  applyMatrix3(e) {
    const t = this.x, i = this.y, s = e.elements;
    return this.x = s[0] * t + s[3] * i + s[6], this.y = s[1] * t + s[4] * i + s[7], this;
  }
  /**
   * If this vector's x or y value is greater than the given vector's x or y
   * value, replace that value with the corresponding min value.
   *
   * @param {Vector2} v - The vector.
   * @return {Vector2} A reference to this vector.
   */
  min(e) {
    return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this;
  }
  /**
   * If this vector's x or y value is less than the given vector's x or y
   * value, replace that value with the corresponding max value.
   *
   * @param {Vector2} v - The vector.
   * @return {Vector2} A reference to this vector.
   */
  max(e) {
    return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this;
  }
  /**
   * If this vector's x or y value is greater than the max vector's x or y
   * value, it is replaced by the corresponding value.
   * If this vector's x or y value is less than the min vector's x or y value,
   * it is replaced by the corresponding value.
   *
   * @param {Vector2} min - The minimum x and y values.
   * @param {Vector2} max - The maximum x and y values in the desired range.
   * @return {Vector2} A reference to this vector.
   */
  clamp(e, t) {
    return this.x = je(this.x, e.x, t.x), this.y = je(this.y, e.y, t.y), this;
  }
  /**
   * If this vector's x or y values are greater than the max value, they are
   * replaced by the max value.
   * If this vector's x or y values are less than the min value, they are
   * replaced by the min value.
   *
   * @param {number} minVal - The minimum value the components will be clamped to.
   * @param {number} maxVal - The maximum value the components will be clamped to.
   * @return {Vector2} A reference to this vector.
   */
  clampScalar(e, t) {
    return this.x = je(this.x, e, t), this.y = je(this.y, e, t), this;
  }
  /**
   * If this vector's length is greater than the max value, it is replaced by
   * the max value.
   * If this vector's length is less than the min value, it is replaced by the
   * min value.
   *
   * @param {number} min - The minimum value the vector length will be clamped to.
   * @param {number} max - The maximum value the vector length will be clamped to.
   * @return {Vector2} A reference to this vector.
   */
  clampLength(e, t) {
    const i = this.length();
    return this.divideScalar(i || 1).multiplyScalar(je(i, e, t));
  }
  /**
   * The components of this vector are rounded down to the nearest integer value.
   *
   * @return {Vector2} A reference to this vector.
   */
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this;
  }
  /**
   * The components of this vector are rounded up to the nearest integer value.
   *
   * @return {Vector2} A reference to this vector.
   */
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this;
  }
  /**
   * The components of this vector are rounded to the nearest integer value
   *
   * @return {Vector2} A reference to this vector.
   */
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this;
  }
  /**
   * The components of this vector are rounded towards zero (up if negative,
   * down if positive) to an integer value.
   *
   * @return {Vector2} A reference to this vector.
   */
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this;
  }
  /**
   * Inverts this vector - i.e. sets x = -x and y = -y.
   *
   * @return {Vector2} A reference to this vector.
   */
  negate() {
    return this.x = -this.x, this.y = -this.y, this;
  }
  /**
   * Calculates the dot product of the given vector with this instance.
   *
   * @param {Vector2} v - The vector to compute the dot product with.
   * @return {number} The result of the dot product.
   */
  dot(e) {
    return this.x * e.x + this.y * e.y;
  }
  /**
   * Calculates the cross product of the given vector with this instance.
   *
   * @param {Vector2} v - The vector to compute the cross product with.
   * @return {number} The result of the cross product.
   */
  cross(e) {
    return this.x * e.y - this.y * e.x;
  }
  /**
   * Computes the square of the Euclidean length (straight-line length) from
   * (0, 0) to (x, y). If you are comparing the lengths of vectors, you should
   * compare the length squared instead as it is slightly more efficient to calculate.
   *
   * @return {number} The square length of this vector.
   */
  lengthSq() {
    return this.x * this.x + this.y * this.y;
  }
  /**
   * Computes the  Euclidean length (straight-line length) from (0, 0) to (x, y).
   *
   * @return {number} The length of this vector.
   */
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }
  /**
   * Computes the Manhattan length of this vector.
   *
   * @return {number} The length of this vector.
   */
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y);
  }
  /**
   * Converts this vector to a unit vector - that is, sets it equal to a vector
   * with the same direction as this one, but with a vector length of `1`.
   *
   * @return {Vector2} A reference to this vector.
   */
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  /**
   * Computes the angle in radians of this vector with respect to the positive x-axis.
   *
   * @return {number} The angle in radians.
   */
  angle() {
    return Math.atan2(-this.y, -this.x) + Math.PI;
  }
  /**
   * Returns the angle between the given vector and this instance in radians.
   *
   * @param {Vector2} v - The vector to compute the angle with.
   * @return {number} The angle in radians.
   */
  angleTo(e) {
    const t = Math.sqrt(this.lengthSq() * e.lengthSq());
    if (t === 0) return Math.PI / 2;
    const i = this.dot(e) / t;
    return Math.acos(je(i, -1, 1));
  }
  /**
   * Computes the distance from the given vector to this instance.
   *
   * @param {Vector2} v - The vector to compute the distance to.
   * @return {number} The distance.
   */
  distanceTo(e) {
    return Math.sqrt(this.distanceToSquared(e));
  }
  /**
   * Computes the squared distance from the given vector to this instance.
   * If you are just comparing the distance with another distance, you should compare
   * the distance squared instead as it is slightly more efficient to calculate.
   *
   * @param {Vector2} v - The vector to compute the squared distance to.
   * @return {number} The squared distance.
   */
  distanceToSquared(e) {
    const t = this.x - e.x, i = this.y - e.y;
    return t * t + i * i;
  }
  /**
   * Computes the Manhattan distance from the given vector to this instance.
   *
   * @param {Vector2} v - The vector to compute the Manhattan distance to.
   * @return {number} The Manhattan distance.
   */
  manhattanDistanceTo(e) {
    return Math.abs(this.x - e.x) + Math.abs(this.y - e.y);
  }
  /**
   * Sets this vector to a vector with the same direction as this one, but
   * with the specified length.
   *
   * @param {number} length - The new length of this vector.
   * @return {Vector2} A reference to this vector.
   */
  setLength(e) {
    return this.normalize().multiplyScalar(e);
  }
  /**
   * Linearly interpolates between the given vector and this instance, where
   * alpha is the percent distance along the line - alpha = 0 will be this
   * vector, and alpha = 1 will be the given one.
   *
   * @param {Vector2} v - The vector to interpolate towards.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector2} A reference to this vector.
   */
  lerp(e, t) {
    return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this;
  }
  /**
   * Linearly interpolates between the given vectors, where alpha is the percent
   * distance along the line - alpha = 0 will be first vector, and alpha = 1 will
   * be the second one. The result is stored in this instance.
   *
   * @param {Vector2} v1 - The first vector.
   * @param {Vector2} v2 - The second vector.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector2} A reference to this vector.
   */
  lerpVectors(e, t, i) {
    return this.x = e.x + (t.x - e.x) * i, this.y = e.y + (t.y - e.y) * i, this;
  }
  /**
   * Returns `true` if this vector is equal with the given one.
   *
   * @param {Vector2} v - The vector to test for equality.
   * @return {boolean} Whether this vector is equal with the given one.
   */
  equals(e) {
    return e.x === this.x && e.y === this.y;
  }
  /**
   * Sets this vector's x value to be `array[ offset ]` and y
   * value to be `array[ offset + 1 ]`.
   *
   * @param {Array<number>} array - An array holding the vector component values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Vector2} A reference to this vector.
   */
  fromArray(e, t = 0) {
    return this.x = e[t], this.y = e[t + 1], this;
  }
  /**
   * Writes the components of this vector to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the vector components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The vector components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this.x, e[t + 1] = this.y, e;
  }
  /**
   * Sets the components of this vector from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding vector data.
   * @param {number} index - The index into the attribute.
   * @return {Vector2} A reference to this vector.
   */
  fromBufferAttribute(e, t) {
    return this.x = e.getX(t), this.y = e.getY(t), this;
  }
  /**
   * Rotates this vector around the given center by the given angle.
   *
   * @param {Vector2} center - The point around which to rotate.
   * @param {number} angle - The angle to rotate, in radians.
   * @return {Vector2} A reference to this vector.
   */
  rotateAround(e, t) {
    const i = Math.cos(t), s = Math.sin(t), r = this.x - e.x, o = this.y - e.y;
    return this.x = r * i - o * s + e.x, this.y = r * s + o * i + e.y, this;
  }
  /**
   * Sets each component of this vector to a pseudo-random value between `0` and
   * `1`, excluding `1`.
   *
   * @return {Vector2} A reference to this vector.
   */
  random() {
    return this.x = Math.random(), this.y = Math.random(), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y;
  }
}
class Yi {
  /**
   * Constructs a new quaternion.
   *
   * @param {number} [x=0] - The x value of this quaternion.
   * @param {number} [y=0] - The y value of this quaternion.
   * @param {number} [z=0] - The z value of this quaternion.
   * @param {number} [w=1] - The w value of this quaternion.
   */
  constructor(e = 0, t = 0, i = 0, s = 1) {
    this.isQuaternion = !0, this._x = e, this._y = t, this._z = i, this._w = s;
  }
  /**
   * Interpolates between two quaternions via SLERP. This implementation assumes the
   * quaternion data are managed  in flat arrays.
   *
   * @param {Array<number>} dst - The destination array.
   * @param {number} dstOffset - An offset into the destination array.
   * @param {Array<number>} src0 - The source array of the first quaternion.
   * @param {number} srcOffset0 - An offset into the first source array.
   * @param {Array<number>} src1 -  The source array of the second quaternion.
   * @param {number} srcOffset1 - An offset into the second source array.
   * @param {number} t - The interpolation factor in the range `[0,1]`.
   * @see {@link Quaternion#slerp}
   */
  static slerpFlat(e, t, i, s, r, o, a) {
    let l = i[s + 0], c = i[s + 1], u = i[s + 2], h = i[s + 3];
    const f = r[o + 0], p = r[o + 1], v = r[o + 2], x = r[o + 3];
    if (a === 0) {
      e[t + 0] = l, e[t + 1] = c, e[t + 2] = u, e[t + 3] = h;
      return;
    }
    if (a === 1) {
      e[t + 0] = f, e[t + 1] = p, e[t + 2] = v, e[t + 3] = x;
      return;
    }
    if (h !== x || l !== f || c !== p || u !== v) {
      let m = 1 - a;
      const d = l * f + c * p + u * v + h * x, b = d >= 0 ? 1 : -1, A = 1 - d * d;
      if (A > Number.EPSILON) {
        const R = Math.sqrt(A), w = Math.atan2(R, d * b);
        m = Math.sin(m * w) / R, a = Math.sin(a * w) / R;
      }
      const M = a * b;
      if (l = l * m + f * M, c = c * m + p * M, u = u * m + v * M, h = h * m + x * M, m === 1 - a) {
        const R = 1 / Math.sqrt(l * l + c * c + u * u + h * h);
        l *= R, c *= R, u *= R, h *= R;
      }
    }
    e[t] = l, e[t + 1] = c, e[t + 2] = u, e[t + 3] = h;
  }
  /**
   * Multiplies two quaternions. This implementation assumes the quaternion data are managed
   * in flat arrays.
   *
   * @param {Array<number>} dst - The destination array.
   * @param {number} dstOffset - An offset into the destination array.
   * @param {Array<number>} src0 - The source array of the first quaternion.
   * @param {number} srcOffset0 - An offset into the first source array.
   * @param {Array<number>} src1 -  The source array of the second quaternion.
   * @param {number} srcOffset1 - An offset into the second source array.
   * @return {Array<number>} The destination array.
   * @see {@link Quaternion#multiplyQuaternions}.
   */
  static multiplyQuaternionsFlat(e, t, i, s, r, o) {
    const a = i[s], l = i[s + 1], c = i[s + 2], u = i[s + 3], h = r[o], f = r[o + 1], p = r[o + 2], v = r[o + 3];
    return e[t] = a * v + u * h + l * p - c * f, e[t + 1] = l * v + u * f + c * h - a * p, e[t + 2] = c * v + u * p + a * f - l * h, e[t + 3] = u * v - a * h - l * f - c * p, e;
  }
  /**
   * The x value of this quaternion.
   *
   * @type {number}
   * @default 0
   */
  get x() {
    return this._x;
  }
  set x(e) {
    this._x = e, this._onChangeCallback();
  }
  /**
   * The y value of this quaternion.
   *
   * @type {number}
   * @default 0
   */
  get y() {
    return this._y;
  }
  set y(e) {
    this._y = e, this._onChangeCallback();
  }
  /**
   * The z value of this quaternion.
   *
   * @type {number}
   * @default 0
   */
  get z() {
    return this._z;
  }
  set z(e) {
    this._z = e, this._onChangeCallback();
  }
  /**
   * The w value of this quaternion.
   *
   * @type {number}
   * @default 1
   */
  get w() {
    return this._w;
  }
  set w(e) {
    this._w = e, this._onChangeCallback();
  }
  /**
   * Sets the quaternion components.
   *
   * @param {number} x - The x value of this quaternion.
   * @param {number} y - The y value of this quaternion.
   * @param {number} z - The z value of this quaternion.
   * @param {number} w - The w value of this quaternion.
   * @return {Quaternion} A reference to this quaternion.
   */
  set(e, t, i, s) {
    return this._x = e, this._y = t, this._z = i, this._w = s, this._onChangeCallback(), this;
  }
  /**
   * Returns a new quaternion with copied values from this instance.
   *
   * @return {Quaternion} A clone of this instance.
   */
  clone() {
    return new this.constructor(this._x, this._y, this._z, this._w);
  }
  /**
   * Copies the values of the given quaternion to this instance.
   *
   * @param {Quaternion} quaternion - The quaternion to copy.
   * @return {Quaternion} A reference to this quaternion.
   */
  copy(e) {
    return this._x = e.x, this._y = e.y, this._z = e.z, this._w = e.w, this._onChangeCallback(), this;
  }
  /**
   * Sets this quaternion from the rotation specified by the given
   * Euler angles.
   *
   * @param {Euler} euler - The Euler angles.
   * @param {boolean} [update=true] - Whether the internal `onChange` callback should be executed or not.
   * @return {Quaternion} A reference to this quaternion.
   */
  setFromEuler(e, t = !0) {
    const i = e._x, s = e._y, r = e._z, o = e._order, a = Math.cos, l = Math.sin, c = a(i / 2), u = a(s / 2), h = a(r / 2), f = l(i / 2), p = l(s / 2), v = l(r / 2);
    switch (o) {
      case "XYZ":
        this._x = f * u * h + c * p * v, this._y = c * p * h - f * u * v, this._z = c * u * v + f * p * h, this._w = c * u * h - f * p * v;
        break;
      case "YXZ":
        this._x = f * u * h + c * p * v, this._y = c * p * h - f * u * v, this._z = c * u * v - f * p * h, this._w = c * u * h + f * p * v;
        break;
      case "ZXY":
        this._x = f * u * h - c * p * v, this._y = c * p * h + f * u * v, this._z = c * u * v + f * p * h, this._w = c * u * h - f * p * v;
        break;
      case "ZYX":
        this._x = f * u * h - c * p * v, this._y = c * p * h + f * u * v, this._z = c * u * v - f * p * h, this._w = c * u * h + f * p * v;
        break;
      case "YZX":
        this._x = f * u * h + c * p * v, this._y = c * p * h + f * u * v, this._z = c * u * v - f * p * h, this._w = c * u * h - f * p * v;
        break;
      case "XZY":
        this._x = f * u * h - c * p * v, this._y = c * p * h - f * u * v, this._z = c * u * v + f * p * h, this._w = c * u * h + f * p * v;
        break;
      default:
        console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: " + o);
    }
    return t === !0 && this._onChangeCallback(), this;
  }
  /**
   * Sets this quaternion from the given axis and angle.
   *
   * @param {Vector3} axis - The normalized axis.
   * @param {number} angle - The angle in radians.
   * @return {Quaternion} A reference to this quaternion.
   */
  setFromAxisAngle(e, t) {
    const i = t / 2, s = Math.sin(i);
    return this._x = e.x * s, this._y = e.y * s, this._z = e.z * s, this._w = Math.cos(i), this._onChangeCallback(), this;
  }
  /**
   * Sets this quaternion from the given rotation matrix.
   *
   * @param {Matrix4} m - A 4x4 matrix of which the upper 3x3 of matrix is a pure rotation matrix (i.e. unscaled).
   * @return {Quaternion} A reference to this quaternion.
   */
  setFromRotationMatrix(e) {
    const t = e.elements, i = t[0], s = t[4], r = t[8], o = t[1], a = t[5], l = t[9], c = t[2], u = t[6], h = t[10], f = i + a + h;
    if (f > 0) {
      const p = 0.5 / Math.sqrt(f + 1);
      this._w = 0.25 / p, this._x = (u - l) * p, this._y = (r - c) * p, this._z = (o - s) * p;
    } else if (i > a && i > h) {
      const p = 2 * Math.sqrt(1 + i - a - h);
      this._w = (u - l) / p, this._x = 0.25 * p, this._y = (s + o) / p, this._z = (r + c) / p;
    } else if (a > h) {
      const p = 2 * Math.sqrt(1 + a - i - h);
      this._w = (r - c) / p, this._x = (s + o) / p, this._y = 0.25 * p, this._z = (l + u) / p;
    } else {
      const p = 2 * Math.sqrt(1 + h - i - a);
      this._w = (o - s) / p, this._x = (r + c) / p, this._y = (l + u) / p, this._z = 0.25 * p;
    }
    return this._onChangeCallback(), this;
  }
  /**
   * Sets this quaternion to the rotation required to rotate the direction vector
   * `vFrom` to the direction vector `vTo`.
   *
   * @param {Vector3} vFrom - The first (normalized) direction vector.
   * @param {Vector3} vTo - The second (normalized) direction vector.
   * @return {Quaternion} A reference to this quaternion.
   */
  setFromUnitVectors(e, t) {
    let i = e.dot(t) + 1;
    return i < 1e-8 ? (i = 0, Math.abs(e.x) > Math.abs(e.z) ? (this._x = -e.y, this._y = e.x, this._z = 0, this._w = i) : (this._x = 0, this._y = -e.z, this._z = e.y, this._w = i)) : (this._x = e.y * t.z - e.z * t.y, this._y = e.z * t.x - e.x * t.z, this._z = e.x * t.y - e.y * t.x, this._w = i), this.normalize();
  }
  /**
   * Returns the angle between this quaternion and the given one in radians.
   *
   * @param {Quaternion} q - The quaternion to compute the angle with.
   * @return {number} The angle in radians.
   */
  angleTo(e) {
    return 2 * Math.acos(Math.abs(je(this.dot(e), -1, 1)));
  }
  /**
   * Rotates this quaternion by a given angular step to the given quaternion.
   * The method ensures that the final quaternion will not overshoot `q`.
   *
   * @param {Quaternion} q - The target quaternion.
   * @param {number} step - The angular step in radians.
   * @return {Quaternion} A reference to this quaternion.
   */
  rotateTowards(e, t) {
    const i = this.angleTo(e);
    if (i === 0) return this;
    const s = Math.min(1, t / i);
    return this.slerp(e, s), this;
  }
  /**
   * Sets this quaternion to the identity quaternion; that is, to the
   * quaternion that represents "no rotation".
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  identity() {
    return this.set(0, 0, 0, 1);
  }
  /**
   * Inverts this quaternion via {@link Quaternion#conjugate}. The
   * quaternion is assumed to have unit length.
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  invert() {
    return this.conjugate();
  }
  /**
   * Returns the rotational conjugate of this quaternion. The conjugate of a
   * quaternion represents the same rotation in the opposite direction about
   * the rotational axis.
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  conjugate() {
    return this._x *= -1, this._y *= -1, this._z *= -1, this._onChangeCallback(), this;
  }
  /**
   * Calculates the dot product of this quaternion and the given one.
   *
   * @param {Quaternion} v - The quaternion to compute the dot product with.
   * @return {number} The result of the dot product.
   */
  dot(e) {
    return this._x * e._x + this._y * e._y + this._z * e._z + this._w * e._w;
  }
  /**
   * Computes the squared Euclidean length (straight-line length) of this quaternion,
   * considered as a 4 dimensional vector. This can be useful if you are comparing the
   * lengths of two quaternions, as this is a slightly more efficient calculation than
   * {@link Quaternion#length}.
   *
   * @return {number} The squared Euclidean length.
   */
  lengthSq() {
    return this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w;
  }
  /**
   * Computes the Euclidean length (straight-line length) of this quaternion,
   * considered as a 4 dimensional vector.
   *
   * @return {number} The Euclidean length.
   */
  length() {
    return Math.sqrt(this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w);
  }
  /**
   * Normalizes this quaternion - that is, calculated the quaternion that performs
   * the same rotation as this one, but has a length equal to `1`.
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  normalize() {
    let e = this.length();
    return e === 0 ? (this._x = 0, this._y = 0, this._z = 0, this._w = 1) : (e = 1 / e, this._x = this._x * e, this._y = this._y * e, this._z = this._z * e, this._w = this._w * e), this._onChangeCallback(), this;
  }
  /**
   * Multiplies this quaternion by the given one.
   *
   * @param {Quaternion} q - The quaternion.
   * @return {Quaternion} A reference to this quaternion.
   */
  multiply(e) {
    return this.multiplyQuaternions(this, e);
  }
  /**
   * Pre-multiplies this quaternion by the given one.
   *
   * @param {Quaternion} q - The quaternion.
   * @return {Quaternion} A reference to this quaternion.
   */
  premultiply(e) {
    return this.multiplyQuaternions(e, this);
  }
  /**
   * Multiplies the given quaternions and stores the result in this instance.
   *
   * @param {Quaternion} a - The first quaternion.
   * @param {Quaternion} b - The second quaternion.
   * @return {Quaternion} A reference to this quaternion.
   */
  multiplyQuaternions(e, t) {
    const i = e._x, s = e._y, r = e._z, o = e._w, a = t._x, l = t._y, c = t._z, u = t._w;
    return this._x = i * u + o * a + s * c - r * l, this._y = s * u + o * l + r * a - i * c, this._z = r * u + o * c + i * l - s * a, this._w = o * u - i * a - s * l - r * c, this._onChangeCallback(), this;
  }
  /**
   * Performs a spherical linear interpolation between quaternions.
   *
   * @param {Quaternion} qb - The target quaternion.
   * @param {number} t - The interpolation factor in the closed interval `[0, 1]`.
   * @return {Quaternion} A reference to this quaternion.
   */
  slerp(e, t) {
    if (t === 0) return this;
    if (t === 1) return this.copy(e);
    const i = this._x, s = this._y, r = this._z, o = this._w;
    let a = o * e._w + i * e._x + s * e._y + r * e._z;
    if (a < 0 ? (this._w = -e._w, this._x = -e._x, this._y = -e._y, this._z = -e._z, a = -a) : this.copy(e), a >= 1)
      return this._w = o, this._x = i, this._y = s, this._z = r, this;
    const l = 1 - a * a;
    if (l <= Number.EPSILON) {
      const p = 1 - t;
      return this._w = p * o + t * this._w, this._x = p * i + t * this._x, this._y = p * s + t * this._y, this._z = p * r + t * this._z, this.normalize(), this;
    }
    const c = Math.sqrt(l), u = Math.atan2(c, a), h = Math.sin((1 - t) * u) / c, f = Math.sin(t * u) / c;
    return this._w = o * h + this._w * f, this._x = i * h + this._x * f, this._y = s * h + this._y * f, this._z = r * h + this._z * f, this._onChangeCallback(), this;
  }
  /**
   * Performs a spherical linear interpolation between the given quaternions
   * and stores the result in this quaternion.
   *
   * @param {Quaternion} qa - The source quaternion.
   * @param {Quaternion} qb - The target quaternion.
   * @param {number} t - The interpolation factor in the closed interval `[0, 1]`.
   * @return {Quaternion} A reference to this quaternion.
   */
  slerpQuaternions(e, t, i) {
    return this.copy(e).slerp(t, i);
  }
  /**
   * Sets this quaternion to a uniformly random, normalized quaternion.
   *
   * @return {Quaternion} A reference to this quaternion.
   */
  random() {
    const e = 2 * Math.PI * Math.random(), t = 2 * Math.PI * Math.random(), i = Math.random(), s = Math.sqrt(1 - i), r = Math.sqrt(i);
    return this.set(
      s * Math.sin(e),
      s * Math.cos(e),
      r * Math.sin(t),
      r * Math.cos(t)
    );
  }
  /**
   * Returns `true` if this quaternion is equal with the given one.
   *
   * @param {Quaternion} quaternion - The quaternion to test for equality.
   * @return {boolean} Whether this quaternion is equal with the given one.
   */
  equals(e) {
    return e._x === this._x && e._y === this._y && e._z === this._z && e._w === this._w;
  }
  /**
   * Sets this quaternion's components from the given array.
   *
   * @param {Array<number>} array - An array holding the quaternion component values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Quaternion} A reference to this quaternion.
   */
  fromArray(e, t = 0) {
    return this._x = e[t], this._y = e[t + 1], this._z = e[t + 2], this._w = e[t + 3], this._onChangeCallback(), this;
  }
  /**
   * Writes the components of this quaternion to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the quaternion components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The quaternion components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this._x, e[t + 1] = this._y, e[t + 2] = this._z, e[t + 3] = this._w, e;
  }
  /**
   * Sets the components of this quaternion from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding quaternion data.
   * @param {number} index - The index into the attribute.
   * @return {Quaternion} A reference to this quaternion.
   */
  fromBufferAttribute(e, t) {
    return this._x = e.getX(t), this._y = e.getY(t), this._z = e.getZ(t), this._w = e.getW(t), this._onChangeCallback(), this;
  }
  /**
   * This methods defines the serialization result of this class. Returns the
   * numerical elements of this quaternion in an array of format `[x, y, z, w]`.
   *
   * @return {Array<number>} The serialized quaternion.
   */
  toJSON() {
    return this.toArray();
  }
  _onChange(e) {
    return this._onChangeCallback = e, this;
  }
  _onChangeCallback() {
  }
  *[Symbol.iterator]() {
    yield this._x, yield this._y, yield this._z, yield this._w;
  }
}
class N {
  /**
   * Constructs a new 3D vector.
   *
   * @param {number} [x=0] - The x value of this vector.
   * @param {number} [y=0] - The y value of this vector.
   * @param {number} [z=0] - The z value of this vector.
   */
  constructor(e = 0, t = 0, i = 0) {
    N.prototype.isVector3 = !0, this.x = e, this.y = t, this.z = i;
  }
  /**
   * Sets the vector components.
   *
   * @param {number} x - The value of the x component.
   * @param {number} y - The value of the y component.
   * @param {number} z - The value of the z component.
   * @return {Vector3} A reference to this vector.
   */
  set(e, t, i) {
    return i === void 0 && (i = this.z), this.x = e, this.y = t, this.z = i, this;
  }
  /**
   * Sets the vector components to the same value.
   *
   * @param {number} scalar - The value to set for all vector components.
   * @return {Vector3} A reference to this vector.
   */
  setScalar(e) {
    return this.x = e, this.y = e, this.z = e, this;
  }
  /**
   * Sets the vector's x component to the given value
   *
   * @param {number} x - The value to set.
   * @return {Vector3} A reference to this vector.
   */
  setX(e) {
    return this.x = e, this;
  }
  /**
   * Sets the vector's y component to the given value
   *
   * @param {number} y - The value to set.
   * @return {Vector3} A reference to this vector.
   */
  setY(e) {
    return this.y = e, this;
  }
  /**
   * Sets the vector's z component to the given value
   *
   * @param {number} z - The value to set.
   * @return {Vector3} A reference to this vector.
   */
  setZ(e) {
    return this.z = e, this;
  }
  /**
   * Allows to set a vector component with an index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y, `2` equals to z.
   * @param {number} value - The value to set.
   * @return {Vector3} A reference to this vector.
   */
  setComponent(e, t) {
    switch (e) {
      case 0:
        this.x = t;
        break;
      case 1:
        this.y = t;
        break;
      case 2:
        this.z = t;
        break;
      default:
        throw new Error("index is out of range: " + e);
    }
    return this;
  }
  /**
   * Returns the value of the vector component which matches the given index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y, `2` equals to z.
   * @return {number} A vector component value.
   */
  getComponent(e) {
    switch (e) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      default:
        throw new Error("index is out of range: " + e);
    }
  }
  /**
   * Returns a new vector with copied values from this instance.
   *
   * @return {Vector3} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.x, this.y, this.z);
  }
  /**
   * Copies the values of the given vector to this instance.
   *
   * @param {Vector3} v - The vector to copy.
   * @return {Vector3} A reference to this vector.
   */
  copy(e) {
    return this.x = e.x, this.y = e.y, this.z = e.z, this;
  }
  /**
   * Adds the given vector to this instance.
   *
   * @param {Vector3} v - The vector to add.
   * @return {Vector3} A reference to this vector.
   */
  add(e) {
    return this.x += e.x, this.y += e.y, this.z += e.z, this;
  }
  /**
   * Adds the given scalar value to all components of this instance.
   *
   * @param {number} s - The scalar to add.
   * @return {Vector3} A reference to this vector.
   */
  addScalar(e) {
    return this.x += e, this.y += e, this.z += e, this;
  }
  /**
   * Adds the given vectors and stores the result in this instance.
   *
   * @param {Vector3} a - The first vector.
   * @param {Vector3} b - The second vector.
   * @return {Vector3} A reference to this vector.
   */
  addVectors(e, t) {
    return this.x = e.x + t.x, this.y = e.y + t.y, this.z = e.z + t.z, this;
  }
  /**
   * Adds the given vector scaled by the given factor to this instance.
   *
   * @param {Vector3|Vector4} v - The vector.
   * @param {number} s - The factor that scales `v`.
   * @return {Vector3} A reference to this vector.
   */
  addScaledVector(e, t) {
    return this.x += e.x * t, this.y += e.y * t, this.z += e.z * t, this;
  }
  /**
   * Subtracts the given vector from this instance.
   *
   * @param {Vector3} v - The vector to subtract.
   * @return {Vector3} A reference to this vector.
   */
  sub(e) {
    return this.x -= e.x, this.y -= e.y, this.z -= e.z, this;
  }
  /**
   * Subtracts the given scalar value from all components of this instance.
   *
   * @param {number} s - The scalar to subtract.
   * @return {Vector3} A reference to this vector.
   */
  subScalar(e) {
    return this.x -= e, this.y -= e, this.z -= e, this;
  }
  /**
   * Subtracts the given vectors and stores the result in this instance.
   *
   * @param {Vector3} a - The first vector.
   * @param {Vector3} b - The second vector.
   * @return {Vector3} A reference to this vector.
   */
  subVectors(e, t) {
    return this.x = e.x - t.x, this.y = e.y - t.y, this.z = e.z - t.z, this;
  }
  /**
   * Multiplies the given vector with this instance.
   *
   * @param {Vector3} v - The vector to multiply.
   * @return {Vector3} A reference to this vector.
   */
  multiply(e) {
    return this.x *= e.x, this.y *= e.y, this.z *= e.z, this;
  }
  /**
   * Multiplies the given scalar value with all components of this instance.
   *
   * @param {number} scalar - The scalar to multiply.
   * @return {Vector3} A reference to this vector.
   */
  multiplyScalar(e) {
    return this.x *= e, this.y *= e, this.z *= e, this;
  }
  /**
   * Multiplies the given vectors and stores the result in this instance.
   *
   * @param {Vector3} a - The first vector.
   * @param {Vector3} b - The second vector.
   * @return {Vector3} A reference to this vector.
   */
  multiplyVectors(e, t) {
    return this.x = e.x * t.x, this.y = e.y * t.y, this.z = e.z * t.z, this;
  }
  /**
   * Applies the given Euler rotation to this vector.
   *
   * @param {Euler} euler - The Euler angles.
   * @return {Vector3} A reference to this vector.
   */
  applyEuler(e) {
    return this.applyQuaternion(Du.setFromEuler(e));
  }
  /**
   * Applies a rotation specified by an axis and an angle to this vector.
   *
   * @param {Vector3} axis - A normalized vector representing the rotation axis.
   * @param {number} angle - The angle in radians.
   * @return {Vector3} A reference to this vector.
   */
  applyAxisAngle(e, t) {
    return this.applyQuaternion(Du.setFromAxisAngle(e, t));
  }
  /**
   * Multiplies this vector with the given 3x3 matrix.
   *
   * @param {Matrix3} m - The 3x3 matrix.
   * @return {Vector3} A reference to this vector.
   */
  applyMatrix3(e) {
    const t = this.x, i = this.y, s = this.z, r = e.elements;
    return this.x = r[0] * t + r[3] * i + r[6] * s, this.y = r[1] * t + r[4] * i + r[7] * s, this.z = r[2] * t + r[5] * i + r[8] * s, this;
  }
  /**
   * Multiplies this vector by the given normal matrix and normalizes
   * the result.
   *
   * @param {Matrix3} m - The normal matrix.
   * @return {Vector3} A reference to this vector.
   */
  applyNormalMatrix(e) {
    return this.applyMatrix3(e).normalize();
  }
  /**
   * Multiplies this vector (with an implicit 1 in the 4th dimension) by m, and
   * divides by perspective.
   *
   * @param {Matrix4} m - The matrix to apply.
   * @return {Vector3} A reference to this vector.
   */
  applyMatrix4(e) {
    const t = this.x, i = this.y, s = this.z, r = e.elements, o = 1 / (r[3] * t + r[7] * i + r[11] * s + r[15]);
    return this.x = (r[0] * t + r[4] * i + r[8] * s + r[12]) * o, this.y = (r[1] * t + r[5] * i + r[9] * s + r[13]) * o, this.z = (r[2] * t + r[6] * i + r[10] * s + r[14]) * o, this;
  }
  /**
   * Applies the given Quaternion to this vector.
   *
   * @param {Quaternion} q - The Quaternion.
   * @return {Vector3} A reference to this vector.
   */
  applyQuaternion(e) {
    const t = this.x, i = this.y, s = this.z, r = e.x, o = e.y, a = e.z, l = e.w, c = 2 * (o * s - a * i), u = 2 * (a * t - r * s), h = 2 * (r * i - o * t);
    return this.x = t + l * c + o * h - a * u, this.y = i + l * u + a * c - r * h, this.z = s + l * h + r * u - o * c, this;
  }
  /**
   * Projects this vector from world space into the camera's normalized
   * device coordinate (NDC) space.
   *
   * @param {Camera} camera - The camera.
   * @return {Vector3} A reference to this vector.
   */
  project(e) {
    return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix);
  }
  /**
   * Unprojects this vector from the camera's normalized device coordinate (NDC)
   * space into world space.
   *
   * @param {Camera} camera - The camera.
   * @return {Vector3} A reference to this vector.
   */
  unproject(e) {
    return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld);
  }
  /**
   * Transforms the direction of this vector by a matrix (the upper left 3 x 3
   * subset of the given 4x4 matrix and then normalizes the result.
   *
   * @param {Matrix4} m - The matrix.
   * @return {Vector3} A reference to this vector.
   */
  transformDirection(e) {
    const t = this.x, i = this.y, s = this.z, r = e.elements;
    return this.x = r[0] * t + r[4] * i + r[8] * s, this.y = r[1] * t + r[5] * i + r[9] * s, this.z = r[2] * t + r[6] * i + r[10] * s, this.normalize();
  }
  /**
   * Divides this instance by the given vector.
   *
   * @param {Vector3} v - The vector to divide.
   * @return {Vector3} A reference to this vector.
   */
  divide(e) {
    return this.x /= e.x, this.y /= e.y, this.z /= e.z, this;
  }
  /**
   * Divides this vector by the given scalar.
   *
   * @param {number} scalar - The scalar to divide.
   * @return {Vector3} A reference to this vector.
   */
  divideScalar(e) {
    return this.multiplyScalar(1 / e);
  }
  /**
   * If this vector's x, y or z value is greater than the given vector's x, y or z
   * value, replace that value with the corresponding min value.
   *
   * @param {Vector3} v - The vector.
   * @return {Vector3} A reference to this vector.
   */
  min(e) {
    return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this.z = Math.min(this.z, e.z), this;
  }
  /**
   * If this vector's x, y or z value is less than the given vector's x, y or z
   * value, replace that value with the corresponding max value.
   *
   * @param {Vector3} v - The vector.
   * @return {Vector3} A reference to this vector.
   */
  max(e) {
    return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this.z = Math.max(this.z, e.z), this;
  }
  /**
   * If this vector's x, y or z value is greater than the max vector's x, y or z
   * value, it is replaced by the corresponding value.
   * If this vector's x, y or z value is less than the min vector's x, y or z value,
   * it is replaced by the corresponding value.
   *
   * @param {Vector3} min - The minimum x, y and z values.
   * @param {Vector3} max - The maximum x, y and z values in the desired range.
   * @return {Vector3} A reference to this vector.
   */
  clamp(e, t) {
    return this.x = je(this.x, e.x, t.x), this.y = je(this.y, e.y, t.y), this.z = je(this.z, e.z, t.z), this;
  }
  /**
   * If this vector's x, y or z values are greater than the max value, they are
   * replaced by the max value.
   * If this vector's x, y or z values are less than the min value, they are
   * replaced by the min value.
   *
   * @param {number} minVal - The minimum value the components will be clamped to.
   * @param {number} maxVal - The maximum value the components will be clamped to.
   * @return {Vector3} A reference to this vector.
   */
  clampScalar(e, t) {
    return this.x = je(this.x, e, t), this.y = je(this.y, e, t), this.z = je(this.z, e, t), this;
  }
  /**
   * If this vector's length is greater than the max value, it is replaced by
   * the max value.
   * If this vector's length is less than the min value, it is replaced by the
   * min value.
   *
   * @param {number} min - The minimum value the vector length will be clamped to.
   * @param {number} max - The maximum value the vector length will be clamped to.
   * @return {Vector3} A reference to this vector.
   */
  clampLength(e, t) {
    const i = this.length();
    return this.divideScalar(i || 1).multiplyScalar(je(i, e, t));
  }
  /**
   * The components of this vector are rounded down to the nearest integer value.
   *
   * @return {Vector3} A reference to this vector.
   */
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this;
  }
  /**
   * The components of this vector are rounded up to the nearest integer value.
   *
   * @return {Vector3} A reference to this vector.
   */
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this;
  }
  /**
   * The components of this vector are rounded to the nearest integer value
   *
   * @return {Vector3} A reference to this vector.
   */
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this;
  }
  /**
   * The components of this vector are rounded towards zero (up if negative,
   * down if positive) to an integer value.
   *
   * @return {Vector3} A reference to this vector.
   */
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this;
  }
  /**
   * Inverts this vector - i.e. sets x = -x, y = -y and z = -z.
   *
   * @return {Vector3} A reference to this vector.
   */
  negate() {
    return this.x = -this.x, this.y = -this.y, this.z = -this.z, this;
  }
  /**
   * Calculates the dot product of the given vector with this instance.
   *
   * @param {Vector3} v - The vector to compute the dot product with.
   * @return {number} The result of the dot product.
   */
  dot(e) {
    return this.x * e.x + this.y * e.y + this.z * e.z;
  }
  // TODO lengthSquared?
  /**
   * Computes the square of the Euclidean length (straight-line length) from
   * (0, 0, 0) to (x, y, z). If you are comparing the lengths of vectors, you should
   * compare the length squared instead as it is slightly more efficient to calculate.
   *
   * @return {number} The square length of this vector.
   */
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }
  /**
   * Computes the  Euclidean length (straight-line length) from (0, 0, 0) to (x, y, z).
   *
   * @return {number} The length of this vector.
   */
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }
  /**
   * Computes the Manhattan length of this vector.
   *
   * @return {number} The length of this vector.
   */
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z);
  }
  /**
   * Converts this vector to a unit vector - that is, sets it equal to a vector
   * with the same direction as this one, but with a vector length of `1`.
   *
   * @return {Vector3} A reference to this vector.
   */
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  /**
   * Sets this vector to a vector with the same direction as this one, but
   * with the specified length.
   *
   * @param {number} length - The new length of this vector.
   * @return {Vector3} A reference to this vector.
   */
  setLength(e) {
    return this.normalize().multiplyScalar(e);
  }
  /**
   * Linearly interpolates between the given vector and this instance, where
   * alpha is the percent distance along the line - alpha = 0 will be this
   * vector, and alpha = 1 will be the given one.
   *
   * @param {Vector3} v - The vector to interpolate towards.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector3} A reference to this vector.
   */
  lerp(e, t) {
    return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this.z += (e.z - this.z) * t, this;
  }
  /**
   * Linearly interpolates between the given vectors, where alpha is the percent
   * distance along the line - alpha = 0 will be first vector, and alpha = 1 will
   * be the second one. The result is stored in this instance.
   *
   * @param {Vector3} v1 - The first vector.
   * @param {Vector3} v2 - The second vector.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector3} A reference to this vector.
   */
  lerpVectors(e, t, i) {
    return this.x = e.x + (t.x - e.x) * i, this.y = e.y + (t.y - e.y) * i, this.z = e.z + (t.z - e.z) * i, this;
  }
  /**
   * Calculates the cross product of the given vector with this instance.
   *
   * @param {Vector3} v - The vector to compute the cross product with.
   * @return {Vector3} The result of the cross product.
   */
  cross(e) {
    return this.crossVectors(this, e);
  }
  /**
   * Calculates the cross product of the given vectors and stores the result
   * in this instance.
   *
   * @param {Vector3} a - The first vector.
   * @param {Vector3} b - The second vector.
   * @return {Vector3} A reference to this vector.
   */
  crossVectors(e, t) {
    const i = e.x, s = e.y, r = e.z, o = t.x, a = t.y, l = t.z;
    return this.x = s * l - r * a, this.y = r * o - i * l, this.z = i * a - s * o, this;
  }
  /**
   * Projects this vector onto the given one.
   *
   * @param {Vector3} v - The vector to project to.
   * @return {Vector3} A reference to this vector.
   */
  projectOnVector(e) {
    const t = e.lengthSq();
    if (t === 0) return this.set(0, 0, 0);
    const i = e.dot(this) / t;
    return this.copy(e).multiplyScalar(i);
  }
  /**
   * Projects this vector onto a plane by subtracting this
   * vector projected onto the plane's normal from this vector.
   *
   * @param {Vector3} planeNormal - The plane normal.
   * @return {Vector3} A reference to this vector.
   */
  projectOnPlane(e) {
    return Sa.copy(this).projectOnVector(e), this.sub(Sa);
  }
  /**
   * Reflects this vector off a plane orthogonal to the given normal vector.
   *
   * @param {Vector3} normal - The (normalized) normal vector.
   * @return {Vector3} A reference to this vector.
   */
  reflect(e) {
    return this.sub(Sa.copy(e).multiplyScalar(2 * this.dot(e)));
  }
  /**
   * Returns the angle between the given vector and this instance in radians.
   *
   * @param {Vector3} v - The vector to compute the angle with.
   * @return {number} The angle in radians.
   */
  angleTo(e) {
    const t = Math.sqrt(this.lengthSq() * e.lengthSq());
    if (t === 0) return Math.PI / 2;
    const i = this.dot(e) / t;
    return Math.acos(je(i, -1, 1));
  }
  /**
   * Computes the distance from the given vector to this instance.
   *
   * @param {Vector3} v - The vector to compute the distance to.
   * @return {number} The distance.
   */
  distanceTo(e) {
    return Math.sqrt(this.distanceToSquared(e));
  }
  /**
   * Computes the squared distance from the given vector to this instance.
   * If you are just comparing the distance with another distance, you should compare
   * the distance squared instead as it is slightly more efficient to calculate.
   *
   * @param {Vector3} v - The vector to compute the squared distance to.
   * @return {number} The squared distance.
   */
  distanceToSquared(e) {
    const t = this.x - e.x, i = this.y - e.y, s = this.z - e.z;
    return t * t + i * i + s * s;
  }
  /**
   * Computes the Manhattan distance from the given vector to this instance.
   *
   * @param {Vector3} v - The vector to compute the Manhattan distance to.
   * @return {number} The Manhattan distance.
   */
  manhattanDistanceTo(e) {
    return Math.abs(this.x - e.x) + Math.abs(this.y - e.y) + Math.abs(this.z - e.z);
  }
  /**
   * Sets the vector components from the given spherical coordinates.
   *
   * @param {Spherical} s - The spherical coordinates.
   * @return {Vector3} A reference to this vector.
   */
  setFromSpherical(e) {
    return this.setFromSphericalCoords(e.radius, e.phi, e.theta);
  }
  /**
   * Sets the vector components from the given spherical coordinates.
   *
   * @param {number} radius - The radius.
   * @param {number} phi - The phi angle in radians.
   * @param {number} theta - The theta angle in radians.
   * @return {Vector3} A reference to this vector.
   */
  setFromSphericalCoords(e, t, i) {
    const s = Math.sin(t) * e;
    return this.x = s * Math.sin(i), this.y = Math.cos(t) * e, this.z = s * Math.cos(i), this;
  }
  /**
   * Sets the vector components from the given cylindrical coordinates.
   *
   * @param {Cylindrical} c - The cylindrical coordinates.
   * @return {Vector3} A reference to this vector.
   */
  setFromCylindrical(e) {
    return this.setFromCylindricalCoords(e.radius, e.theta, e.y);
  }
  /**
   * Sets the vector components from the given cylindrical coordinates.
   *
   * @param {number} radius - The radius.
   * @param {number} theta - The theta angle in radians.
   * @param {number} y - The y value.
   * @return {Vector3} A reference to this vector.
   */
  setFromCylindricalCoords(e, t, i) {
    return this.x = e * Math.sin(t), this.y = i, this.z = e * Math.cos(t), this;
  }
  /**
   * Sets the vector components to the position elements of the
   * given transformation matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Vector3} A reference to this vector.
   */
  setFromMatrixPosition(e) {
    const t = e.elements;
    return this.x = t[12], this.y = t[13], this.z = t[14], this;
  }
  /**
   * Sets the vector components to the scale elements of the
   * given transformation matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Vector3} A reference to this vector.
   */
  setFromMatrixScale(e) {
    const t = this.setFromMatrixColumn(e, 0).length(), i = this.setFromMatrixColumn(e, 1).length(), s = this.setFromMatrixColumn(e, 2).length();
    return this.x = t, this.y = i, this.z = s, this;
  }
  /**
   * Sets the vector components from the specified matrix column.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @param {number} index - The column index.
   * @return {Vector3} A reference to this vector.
   */
  setFromMatrixColumn(e, t) {
    return this.fromArray(e.elements, t * 4);
  }
  /**
   * Sets the vector components from the specified matrix column.
   *
   * @param {Matrix3} m - The 3x3 matrix.
   * @param {number} index - The column index.
   * @return {Vector3} A reference to this vector.
   */
  setFromMatrix3Column(e, t) {
    return this.fromArray(e.elements, t * 3);
  }
  /**
   * Sets the vector components from the given Euler angles.
   *
   * @param {Euler} e - The Euler angles to set.
   * @return {Vector3} A reference to this vector.
   */
  setFromEuler(e) {
    return this.x = e._x, this.y = e._y, this.z = e._z, this;
  }
  /**
   * Sets the vector components from the RGB components of the
   * given color.
   *
   * @param {Color} c - The color to set.
   * @return {Vector3} A reference to this vector.
   */
  setFromColor(e) {
    return this.x = e.r, this.y = e.g, this.z = e.b, this;
  }
  /**
   * Returns `true` if this vector is equal with the given one.
   *
   * @param {Vector3} v - The vector to test for equality.
   * @return {boolean} Whether this vector is equal with the given one.
   */
  equals(e) {
    return e.x === this.x && e.y === this.y && e.z === this.z;
  }
  /**
   * Sets this vector's x value to be `array[ offset ]`, y value to be `array[ offset + 1 ]`
   * and z value to be `array[ offset + 2 ]`.
   *
   * @param {Array<number>} array - An array holding the vector component values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Vector3} A reference to this vector.
   */
  fromArray(e, t = 0) {
    return this.x = e[t], this.y = e[t + 1], this.z = e[t + 2], this;
  }
  /**
   * Writes the components of this vector to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the vector components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The vector components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this.x, e[t + 1] = this.y, e[t + 2] = this.z, e;
  }
  /**
   * Sets the components of this vector from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding vector data.
   * @param {number} index - The index into the attribute.
   * @return {Vector3} A reference to this vector.
   */
  fromBufferAttribute(e, t) {
    return this.x = e.getX(t), this.y = e.getY(t), this.z = e.getZ(t), this;
  }
  /**
   * Sets each component of this vector to a pseudo-random value between `0` and
   * `1`, excluding `1`.
   *
   * @return {Vector3} A reference to this vector.
   */
  random() {
    return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this;
  }
  /**
   * Sets this vector to a uniformly random point on a unit sphere.
   *
   * @return {Vector3} A reference to this vector.
   */
  randomDirection() {
    const e = Math.random() * Math.PI * 2, t = Math.random() * 2 - 1, i = Math.sqrt(1 - t * t);
    return this.x = i * Math.cos(e), this.y = t, this.z = i * Math.sin(e), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y, yield this.z;
  }
}
const Sa = /* @__PURE__ */ new N(), Du = /* @__PURE__ */ new Yi();
class Ye {
  /**
   * Constructs a new 3x3 matrix. The arguments are supposed to be
   * in row-major order. If no arguments are provided, the constructor
   * initializes the matrix as an identity matrix.
   *
   * @param {number} [n11] - 1-1 matrix element.
   * @param {number} [n12] - 1-2 matrix element.
   * @param {number} [n13] - 1-3 matrix element.
   * @param {number} [n21] - 2-1 matrix element.
   * @param {number} [n22] - 2-2 matrix element.
   * @param {number} [n23] - 2-3 matrix element.
   * @param {number} [n31] - 3-1 matrix element.
   * @param {number} [n32] - 3-2 matrix element.
   * @param {number} [n33] - 3-3 matrix element.
   */
  constructor(e, t, i, s, r, o, a, l, c) {
    Ye.prototype.isMatrix3 = !0, this.elements = [
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ], e !== void 0 && this.set(e, t, i, s, r, o, a, l, c);
  }
  /**
   * Sets the elements of the matrix.The arguments are supposed to be
   * in row-major order.
   *
   * @param {number} [n11] - 1-1 matrix element.
   * @param {number} [n12] - 1-2 matrix element.
   * @param {number} [n13] - 1-3 matrix element.
   * @param {number} [n21] - 2-1 matrix element.
   * @param {number} [n22] - 2-2 matrix element.
   * @param {number} [n23] - 2-3 matrix element.
   * @param {number} [n31] - 3-1 matrix element.
   * @param {number} [n32] - 3-2 matrix element.
   * @param {number} [n33] - 3-3 matrix element.
   * @return {Matrix3} A reference to this matrix.
   */
  set(e, t, i, s, r, o, a, l, c) {
    const u = this.elements;
    return u[0] = e, u[1] = s, u[2] = a, u[3] = t, u[4] = r, u[5] = l, u[6] = i, u[7] = o, u[8] = c, this;
  }
  /**
   * Sets this matrix to the 3x3 identity matrix.
   *
   * @return {Matrix3} A reference to this matrix.
   */
  identity() {
    return this.set(
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Copies the values of the given matrix to this instance.
   *
   * @param {Matrix3} m - The matrix to copy.
   * @return {Matrix3} A reference to this matrix.
   */
  copy(e) {
    const t = this.elements, i = e.elements;
    return t[0] = i[0], t[1] = i[1], t[2] = i[2], t[3] = i[3], t[4] = i[4], t[5] = i[5], t[6] = i[6], t[7] = i[7], t[8] = i[8], this;
  }
  /**
   * Extracts the basis of this matrix into the three axis vectors provided.
   *
   * @param {Vector3} xAxis - The basis's x axis.
   * @param {Vector3} yAxis - The basis's y axis.
   * @param {Vector3} zAxis - The basis's z axis.
   * @return {Matrix3} A reference to this matrix.
   */
  extractBasis(e, t, i) {
    return e.setFromMatrix3Column(this, 0), t.setFromMatrix3Column(this, 1), i.setFromMatrix3Column(this, 2), this;
  }
  /**
   * Set this matrix to the upper 3x3 matrix of the given 4x4 matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Matrix3} A reference to this matrix.
   */
  setFromMatrix4(e) {
    const t = e.elements;
    return this.set(
      t[0],
      t[4],
      t[8],
      t[1],
      t[5],
      t[9],
      t[2],
      t[6],
      t[10]
    ), this;
  }
  /**
   * Post-multiplies this matrix by the given 3x3 matrix.
   *
   * @param {Matrix3} m - The matrix to multiply with.
   * @return {Matrix3} A reference to this matrix.
   */
  multiply(e) {
    return this.multiplyMatrices(this, e);
  }
  /**
   * Pre-multiplies this matrix by the given 3x3 matrix.
   *
   * @param {Matrix3} m - The matrix to multiply with.
   * @return {Matrix3} A reference to this matrix.
   */
  premultiply(e) {
    return this.multiplyMatrices(e, this);
  }
  /**
   * Multiples the given 3x3 matrices and stores the result
   * in this matrix.
   *
   * @param {Matrix3} a - The first matrix.
   * @param {Matrix3} b - The second matrix.
   * @return {Matrix3} A reference to this matrix.
   */
  multiplyMatrices(e, t) {
    const i = e.elements, s = t.elements, r = this.elements, o = i[0], a = i[3], l = i[6], c = i[1], u = i[4], h = i[7], f = i[2], p = i[5], v = i[8], x = s[0], m = s[3], d = s[6], b = s[1], A = s[4], M = s[7], R = s[2], w = s[5], D = s[8];
    return r[0] = o * x + a * b + l * R, r[3] = o * m + a * A + l * w, r[6] = o * d + a * M + l * D, r[1] = c * x + u * b + h * R, r[4] = c * m + u * A + h * w, r[7] = c * d + u * M + h * D, r[2] = f * x + p * b + v * R, r[5] = f * m + p * A + v * w, r[8] = f * d + p * M + v * D, this;
  }
  /**
   * Multiplies every component of the matrix by the given scalar.
   *
   * @param {number} s - The scalar.
   * @return {Matrix3} A reference to this matrix.
   */
  multiplyScalar(e) {
    const t = this.elements;
    return t[0] *= e, t[3] *= e, t[6] *= e, t[1] *= e, t[4] *= e, t[7] *= e, t[2] *= e, t[5] *= e, t[8] *= e, this;
  }
  /**
   * Computes and returns the determinant of this matrix.
   *
   * @return {number} The determinant.
   */
  determinant() {
    const e = this.elements, t = e[0], i = e[1], s = e[2], r = e[3], o = e[4], a = e[5], l = e[6], c = e[7], u = e[8];
    return t * o * u - t * a * c - i * r * u + i * a * l + s * r * c - s * o * l;
  }
  /**
   * Inverts this matrix, using the [analytic method]{@link https://en.wikipedia.org/wiki/Invertible_matrix#Analytic_solution}.
   * You can not invert with a determinant of zero. If you attempt this, the method produces
   * a zero matrix instead.
   *
   * @return {Matrix3} A reference to this matrix.
   */
  invert() {
    const e = this.elements, t = e[0], i = e[1], s = e[2], r = e[3], o = e[4], a = e[5], l = e[6], c = e[7], u = e[8], h = u * o - a * c, f = a * l - u * r, p = c * r - o * l, v = t * h + i * f + s * p;
    if (v === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0);
    const x = 1 / v;
    return e[0] = h * x, e[1] = (s * c - u * i) * x, e[2] = (a * i - s * o) * x, e[3] = f * x, e[4] = (u * t - s * l) * x, e[5] = (s * r - a * t) * x, e[6] = p * x, e[7] = (i * l - c * t) * x, e[8] = (o * t - i * r) * x, this;
  }
  /**
   * Transposes this matrix in place.
   *
   * @return {Matrix3} A reference to this matrix.
   */
  transpose() {
    let e;
    const t = this.elements;
    return e = t[1], t[1] = t[3], t[3] = e, e = t[2], t[2] = t[6], t[6] = e, e = t[5], t[5] = t[7], t[7] = e, this;
  }
  /**
   * Computes the normal matrix which is the inverse transpose of the upper
   * left 3x3 portion of the given 4x4 matrix.
   *
   * @param {Matrix4} matrix4 - The 4x4 matrix.
   * @return {Matrix3} A reference to this matrix.
   */
  getNormalMatrix(e) {
    return this.setFromMatrix4(e).invert().transpose();
  }
  /**
   * Transposes this matrix into the supplied array, and returns itself unchanged.
   *
   * @param {Array<number>} r - An array to store the transposed matrix elements.
   * @return {Matrix3} A reference to this matrix.
   */
  transposeIntoArray(e) {
    const t = this.elements;
    return e[0] = t[0], e[1] = t[3], e[2] = t[6], e[3] = t[1], e[4] = t[4], e[5] = t[7], e[6] = t[2], e[7] = t[5], e[8] = t[8], this;
  }
  /**
   * Sets the UV transform matrix from offset, repeat, rotation, and center.
   *
   * @param {number} tx - Offset x.
   * @param {number} ty - Offset y.
   * @param {number} sx - Repeat x.
   * @param {number} sy - Repeat y.
   * @param {number} rotation - Rotation, in radians. Positive values rotate counterclockwise.
   * @param {number} cx - Center x of rotation.
   * @param {number} cy - Center y of rotation
   * @return {Matrix3} A reference to this matrix.
   */
  setUvTransform(e, t, i, s, r, o, a) {
    const l = Math.cos(r), c = Math.sin(r);
    return this.set(
      i * l,
      i * c,
      -i * (l * o + c * a) + o + e,
      -s * c,
      s * l,
      -s * (-c * o + l * a) + a + t,
      0,
      0,
      1
    ), this;
  }
  /**
   * Scales this matrix with the given scalar values.
   *
   * @param {number} sx - The amount to scale in the X axis.
   * @param {number} sy - The amount to scale in the Y axis.
   * @return {Matrix3} A reference to this matrix.
   */
  scale(e, t) {
    return this.premultiply(ya.makeScale(e, t)), this;
  }
  /**
   * Rotates this matrix by the given angle.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix3} A reference to this matrix.
   */
  rotate(e) {
    return this.premultiply(ya.makeRotation(-e)), this;
  }
  /**
   * Translates this matrix by the given scalar values.
   *
   * @param {number} tx - The amount to translate in the X axis.
   * @param {number} ty - The amount to translate in the Y axis.
   * @return {Matrix3} A reference to this matrix.
   */
  translate(e, t) {
    return this.premultiply(ya.makeTranslation(e, t)), this;
  }
  // for 2D Transforms
  /**
   * Sets this matrix as a 2D translation transform.
   *
   * @param {number|Vector2} x - The amount to translate in the X axis or alternatively a translation vector.
   * @param {number} y - The amount to translate in the Y axis.
   * @return {Matrix3} A reference to this matrix.
   */
  makeTranslation(e, t) {
    return e.isVector2 ? this.set(
      1,
      0,
      e.x,
      0,
      1,
      e.y,
      0,
      0,
      1
    ) : this.set(
      1,
      0,
      e,
      0,
      1,
      t,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a 2D rotational transformation.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix3} A reference to this matrix.
   */
  makeRotation(e) {
    const t = Math.cos(e), i = Math.sin(e);
    return this.set(
      t,
      -i,
      0,
      i,
      t,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a 2D scale transform.
   *
   * @param {number} x - The amount to scale in the X axis.
   * @param {number} y - The amount to scale in the Y axis.
   * @return {Matrix3} A reference to this matrix.
   */
  makeScale(e, t) {
    return this.set(
      e,
      0,
      0,
      0,
      t,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Returns `true` if this matrix is equal with the given one.
   *
   * @param {Matrix3} matrix - The matrix to test for equality.
   * @return {boolean} Whether this matrix is equal with the given one.
   */
  equals(e) {
    const t = this.elements, i = e.elements;
    for (let s = 0; s < 9; s++)
      if (t[s] !== i[s]) return !1;
    return !0;
  }
  /**
   * Sets the elements of the matrix from the given array.
   *
   * @param {Array<number>} array - The matrix elements in column-major order.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Matrix3} A reference to this matrix.
   */
  fromArray(e, t = 0) {
    for (let i = 0; i < 9; i++)
      this.elements[i] = e[i + t];
    return this;
  }
  /**
   * Writes the elements of this matrix to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the matrix elements in column-major order.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The matrix elements in column-major order.
   */
  toArray(e = [], t = 0) {
    const i = this.elements;
    return e[t] = i[0], e[t + 1] = i[1], e[t + 2] = i[2], e[t + 3] = i[3], e[t + 4] = i[4], e[t + 5] = i[5], e[t + 6] = i[6], e[t + 7] = i[7], e[t + 8] = i[8], e;
  }
  /**
   * Returns a matrix with copied values from this instance.
   *
   * @return {Matrix3} A clone of this instance.
   */
  clone() {
    return new this.constructor().fromArray(this.elements);
  }
}
const ya = /* @__PURE__ */ new Ye();
function fd(n) {
  for (let e = n.length - 1; e >= 0; --e)
    if (n[e] >= 65535) return !0;
  return !1;
}
function Ho(n) {
  return document.createElementNS("http://www.w3.org/1999/xhtml", n);
}
function Cg() {
  const n = Ho("canvas");
  return n.style.display = "block", n;
}
const Lu = {};
function Rr(n) {
  n in Lu || (Lu[n] = !0, console.warn(n));
}
function Pg(n, e, t) {
  return new Promise(function(i, s) {
    function r() {
      switch (n.clientWaitSync(e, n.SYNC_FLUSH_COMMANDS_BIT, 0)) {
        case n.WAIT_FAILED:
          s();
          break;
        case n.TIMEOUT_EXPIRED:
          setTimeout(r, t);
          break;
        default:
          i();
      }
    }
    setTimeout(r, t);
  });
}
const Iu = /* @__PURE__ */ new Ye().set(
  0.4123908,
  0.3575843,
  0.1804808,
  0.212639,
  0.7151687,
  0.0721923,
  0.0193308,
  0.1191948,
  0.9505322
), Uu = /* @__PURE__ */ new Ye().set(
  3.2409699,
  -1.5373832,
  -0.4986108,
  -0.9692436,
  1.8759675,
  0.0415551,
  0.0556301,
  -0.203977,
  1.0569715
);
function Dg() {
  const n = {
    enabled: !0,
    workingColorSpace: Os,
    /**
     * Implementations of supported color spaces.
     *
     * Required:
     *	- primaries: chromaticity coordinates [ rx ry gx gy bx by ]
     *	- whitePoint: reference white [ x y ]
     *	- transfer: transfer function (pre-defined)
     *	- toXYZ: Matrix3 RGB to XYZ transform
     *	- fromXYZ: Matrix3 XYZ to RGB transform
     *	- luminanceCoefficients: RGB luminance coefficients
     *
     * Optional:
     *  - outputColorSpaceConfig: { drawingBufferColorSpace: ColorSpace, toneMappingMode: 'extended' | 'standard' }
     *  - workingColorSpaceConfig: { unpackColorSpace: ColorSpace }
     *
     * Reference:
     * - https://www.russellcottrell.com/photo/matrixCalculator.htm
     */
    spaces: {},
    convert: function(s, r, o) {
      return this.enabled === !1 || r === o || !r || !o || (this.spaces[r].transfer === ot && (s.r = ti(s.r), s.g = ti(s.g), s.b = ti(s.b)), this.spaces[r].primaries !== this.spaces[o].primaries && (s.applyMatrix3(this.spaces[r].toXYZ), s.applyMatrix3(this.spaces[o].fromXYZ)), this.spaces[o].transfer === ot && (s.r = Ls(s.r), s.g = Ls(s.g), s.b = Ls(s.b))), s;
    },
    workingToColorSpace: function(s, r) {
      return this.convert(s, this.workingColorSpace, r);
    },
    colorSpaceToWorking: function(s, r) {
      return this.convert(s, r, this.workingColorSpace);
    },
    getPrimaries: function(s) {
      return this.spaces[s].primaries;
    },
    getTransfer: function(s) {
      return s === gi ? Bo : this.spaces[s].transfer;
    },
    getToneMappingMode: function(s) {
      return this.spaces[s].outputColorSpaceConfig.toneMappingMode || "standard";
    },
    getLuminanceCoefficients: function(s, r = this.workingColorSpace) {
      return s.fromArray(this.spaces[r].luminanceCoefficients);
    },
    define: function(s) {
      Object.assign(this.spaces, s);
    },
    // Internal APIs
    _getMatrix: function(s, r, o) {
      return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ);
    },
    _getDrawingBufferColorSpace: function(s) {
      return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace;
    },
    _getUnpackColorSpace: function(s = this.workingColorSpace) {
      return this.spaces[s].workingColorSpaceConfig.unpackColorSpace;
    },
    // Deprecated
    fromWorkingColorSpace: function(s, r) {
      return Rr("THREE.ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."), n.workingToColorSpace(s, r);
    },
    toWorkingColorSpace: function(s, r) {
      return Rr("THREE.ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."), n.colorSpaceToWorking(s, r);
    }
  }, e = [0.64, 0.33, 0.3, 0.6, 0.15, 0.06], t = [0.2126, 0.7152, 0.0722], i = [0.3127, 0.329];
  return n.define({
    [Os]: {
      primaries: e,
      whitePoint: i,
      transfer: Bo,
      toXYZ: Iu,
      fromXYZ: Uu,
      luminanceCoefficients: t,
      workingColorSpaceConfig: { unpackColorSpace: tn },
      outputColorSpaceConfig: { drawingBufferColorSpace: tn }
    },
    [tn]: {
      primaries: e,
      whitePoint: i,
      transfer: ot,
      toXYZ: Iu,
      fromXYZ: Uu,
      luminanceCoefficients: t,
      outputColorSpaceConfig: { drawingBufferColorSpace: tn }
    }
  }), n;
}
const Qe = /* @__PURE__ */ Dg();
function ti(n) {
  return n < 0.04045 ? n * 0.0773993808 : Math.pow(n * 0.9478672986 + 0.0521327014, 2.4);
}
function Ls(n) {
  return n < 31308e-7 ? n * 12.92 : 1.055 * Math.pow(n, 0.41666) - 0.055;
}
let is;
class Lg {
  /**
   * Returns a data URI containing a representation of the given image.
   *
   * @param {(HTMLImageElement|HTMLCanvasElement)} image - The image object.
   * @param {string} [type='image/png'] - Indicates the image format.
   * @return {string} The data URI.
   */
  static getDataURL(e, t = "image/png") {
    if (/^data:/i.test(e.src) || typeof HTMLCanvasElement > "u")
      return e.src;
    let i;
    if (e instanceof HTMLCanvasElement)
      i = e;
    else {
      is === void 0 && (is = Ho("canvas")), is.width = e.width, is.height = e.height;
      const s = is.getContext("2d");
      e instanceof ImageData ? s.putImageData(e, 0, 0) : s.drawImage(e, 0, 0, e.width, e.height), i = is;
    }
    return i.toDataURL(t);
  }
  /**
   * Converts the given sRGB image data to linear color space.
   *
   * @param {(HTMLImageElement|HTMLCanvasElement|ImageBitmap|Object)} image - The image object.
   * @return {HTMLCanvasElement|Object} The converted image.
   */
  static sRGBToLinear(e) {
    if (typeof HTMLImageElement < "u" && e instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && e instanceof ImageBitmap) {
      const t = Ho("canvas");
      t.width = e.width, t.height = e.height;
      const i = t.getContext("2d");
      i.drawImage(e, 0, 0, e.width, e.height);
      const s = i.getImageData(0, 0, e.width, e.height), r = s.data;
      for (let o = 0; o < r.length; o++)
        r[o] = ti(r[o] / 255) * 255;
      return i.putImageData(s, 0, 0), t;
    } else if (e.data) {
      const t = e.data.slice(0);
      for (let i = 0; i < t.length; i++)
        t instanceof Uint8Array || t instanceof Uint8ClampedArray ? t[i] = Math.floor(ti(t[i] / 255) * 255) : t[i] = ti(t[i]);
      return {
        data: t,
        width: e.width,
        height: e.height
      };
    } else
      return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."), e;
  }
}
let Ig = 0;
class bc {
  /**
   * Constructs a new video texture.
   *
   * @param {any} [data=null] - The data definition of a texture.
   */
  constructor(e = null) {
    this.isSource = !0, Object.defineProperty(this, "id", { value: Ig++ }), this.uuid = Ir(), this.data = e, this.dataReady = !0, this.version = 0;
  }
  /**
   * Returns the dimensions of the source into the given target vector.
   *
   * @param {(Vector2|Vector3)} target - The target object the result is written into.
   * @return {(Vector2|Vector3)} The dimensions of the source.
   */
  getSize(e) {
    const t = this.data;
    return typeof HTMLVideoElement < "u" && t instanceof HTMLVideoElement ? e.set(t.videoWidth, t.videoHeight, 0) : t instanceof VideoFrame ? e.set(t.displayHeight, t.displayWidth, 0) : t !== null ? e.set(t.width, t.height, t.depth || 0) : e.set(0, 0, 0), e;
  }
  /**
   * When the property is set to `true`, the engine allocates the memory
   * for the texture (if necessary) and triggers the actual texture upload
   * to the GPU next time the source is used.
   *
   * @type {boolean}
   * @default false
   * @param {boolean} value
   */
  set needsUpdate(e) {
    e === !0 && this.version++;
  }
  /**
   * Serializes the source into JSON.
   *
   * @param {?(Object|string)} meta - An optional value holding meta information about the serialization.
   * @return {Object} A JSON object representing the serialized source.
   * @see {@link ObjectLoader#parse}
   */
  toJSON(e) {
    const t = e === void 0 || typeof e == "string";
    if (!t && e.images[this.uuid] !== void 0)
      return e.images[this.uuid];
    const i = {
      uuid: this.uuid,
      url: ""
    }, s = this.data;
    if (s !== null) {
      let r;
      if (Array.isArray(s)) {
        r = [];
        for (let o = 0, a = s.length; o < a; o++)
          s[o].isDataTexture ? r.push(Ea(s[o].image)) : r.push(Ea(s[o]));
      } else
        r = Ea(s);
      i.url = r;
    }
    return t || (e.images[this.uuid] = i), i;
  }
}
function Ea(n) {
  return typeof HTMLImageElement < "u" && n instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && n instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && n instanceof ImageBitmap ? Lg.getDataURL(n) : n.data ? {
    data: Array.from(n.data),
    width: n.width,
    height: n.height,
    type: n.data.constructor.name
  } : (console.warn("THREE.Texture: Unable to serialize Texture."), {});
}
let Ug = 0;
const Ta = /* @__PURE__ */ new N();
class $t extends Zi {
  /**
   * Constructs a new texture.
   *
   * @param {?Object} [image=Texture.DEFAULT_IMAGE] - The image holding the texture data.
   * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
   * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
   * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
   * @param {number} [magFilter=LinearFilter] - The mag filter value.
   * @param {number} [minFilter=LinearMipmapLinearFilter] - The min filter value.
   * @param {number} [format=RGBAFormat] - The texture format.
   * @param {number} [type=UnsignedByteType] - The texture type.
   * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
   * @param {string} [colorSpace=NoColorSpace] - The color space.
   */
  constructor(e = $t.DEFAULT_IMAGE, t = $t.DEFAULT_MAPPING, i = Hi, s = Hi, r = Un, o = Vi, a = xn, l = Bn, c = $t.DEFAULT_ANISOTROPY, u = gi) {
    super(), this.isTexture = !0, Object.defineProperty(this, "id", { value: Ug++ }), this.uuid = Ir(), this.name = "", this.source = new bc(e), this.mipmaps = [], this.mapping = t, this.channel = 0, this.wrapS = i, this.wrapT = s, this.magFilter = r, this.minFilter = o, this.anisotropy = c, this.format = a, this.internalFormat = null, this.type = l, this.offset = new Ve(0, 0), this.repeat = new Ve(1, 1), this.center = new Ve(0, 0), this.rotation = 0, this.matrixAutoUpdate = !0, this.matrix = new Ye(), this.generateMipmaps = !0, this.premultiplyAlpha = !1, this.flipY = !0, this.unpackAlignment = 4, this.colorSpace = u, this.userData = {}, this.updateRanges = [], this.version = 0, this.onUpdate = null, this.renderTarget = null, this.isRenderTargetTexture = !1, this.isArrayTexture = !!(e && e.depth && e.depth > 1), this.pmremVersion = 0;
  }
  /**
   * The width of the texture in pixels.
   */
  get width() {
    return this.source.getSize(Ta).x;
  }
  /**
   * The height of the texture in pixels.
   */
  get height() {
    return this.source.getSize(Ta).y;
  }
  /**
   * The depth of the texture in pixels.
   */
  get depth() {
    return this.source.getSize(Ta).z;
  }
  /**
   * The image object holding the texture data.
   *
   * @type {?Object}
   */
  get image() {
    return this.source.data;
  }
  set image(e = null) {
    this.source.data = e;
  }
  /**
   * Updates the texture transformation matrix from the from the properties {@link Texture#offset},
   * {@link Texture#repeat}, {@link Texture#rotation}, and {@link Texture#center}.
   */
  updateMatrix() {
    this.matrix.setUvTransform(this.offset.x, this.offset.y, this.repeat.x, this.repeat.y, this.rotation, this.center.x, this.center.y);
  }
  /**
   * Adds a range of data in the data texture to be updated on the GPU.
   *
   * @param {number} start - Position at which to start update.
   * @param {number} count - The number of components to update.
   */
  addUpdateRange(e, t) {
    this.updateRanges.push({ start: e, count: t });
  }
  /**
   * Clears the update ranges.
   */
  clearUpdateRanges() {
    this.updateRanges.length = 0;
  }
  /**
   * Returns a new texture with copied values from this instance.
   *
   * @return {Texture} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given texture to this instance.
   *
   * @param {Texture} source - The texture to copy.
   * @return {Texture} A reference to this instance.
   */
  copy(e) {
    return this.name = e.name, this.source = e.source, this.mipmaps = e.mipmaps.slice(0), this.mapping = e.mapping, this.channel = e.channel, this.wrapS = e.wrapS, this.wrapT = e.wrapT, this.magFilter = e.magFilter, this.minFilter = e.minFilter, this.anisotropy = e.anisotropy, this.format = e.format, this.internalFormat = e.internalFormat, this.type = e.type, this.offset.copy(e.offset), this.repeat.copy(e.repeat), this.center.copy(e.center), this.rotation = e.rotation, this.matrixAutoUpdate = e.matrixAutoUpdate, this.matrix.copy(e.matrix), this.generateMipmaps = e.generateMipmaps, this.premultiplyAlpha = e.premultiplyAlpha, this.flipY = e.flipY, this.unpackAlignment = e.unpackAlignment, this.colorSpace = e.colorSpace, this.renderTarget = e.renderTarget, this.isRenderTargetTexture = e.isRenderTargetTexture, this.isArrayTexture = e.isArrayTexture, this.userData = JSON.parse(JSON.stringify(e.userData)), this.needsUpdate = !0, this;
  }
  /**
   * Sets this texture's properties based on `values`.
   * @param {Object} values - A container with texture parameters.
   */
  setValues(e) {
    for (const t in e) {
      const i = e[t];
      if (i === void 0) {
        console.warn(`THREE.Texture.setValues(): parameter '${t}' has value of undefined.`);
        continue;
      }
      const s = this[t];
      if (s === void 0) {
        console.warn(`THREE.Texture.setValues(): property '${t}' does not exist.`);
        continue;
      }
      s && i && s.isVector2 && i.isVector2 || s && i && s.isVector3 && i.isVector3 || s && i && s.isMatrix3 && i.isMatrix3 ? s.copy(i) : this[t] = i;
    }
  }
  /**
   * Serializes the texture into JSON.
   *
   * @param {?(Object|string)} meta - An optional value holding meta information about the serialization.
   * @return {Object} A JSON object representing the serialized texture.
   * @see {@link ObjectLoader#parse}
   */
  toJSON(e) {
    const t = e === void 0 || typeof e == "string";
    if (!t && e.textures[this.uuid] !== void 0)
      return e.textures[this.uuid];
    const i = {
      metadata: {
        version: 4.7,
        type: "Texture",
        generator: "Texture.toJSON"
      },
      uuid: this.uuid,
      name: this.name,
      image: this.source.toJSON(e).uuid,
      mapping: this.mapping,
      channel: this.channel,
      repeat: [this.repeat.x, this.repeat.y],
      offset: [this.offset.x, this.offset.y],
      center: [this.center.x, this.center.y],
      rotation: this.rotation,
      wrap: [this.wrapS, this.wrapT],
      format: this.format,
      internalFormat: this.internalFormat,
      type: this.type,
      colorSpace: this.colorSpace,
      minFilter: this.minFilter,
      magFilter: this.magFilter,
      anisotropy: this.anisotropy,
      flipY: this.flipY,
      generateMipmaps: this.generateMipmaps,
      premultiplyAlpha: this.premultiplyAlpha,
      unpackAlignment: this.unpackAlignment
    };
    return Object.keys(this.userData).length > 0 && (i.userData = this.userData), t || (e.textures[this.uuid] = i), i;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   *
   * @fires Texture#dispose
   */
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
  /**
   * Transforms the given uv vector with the textures uv transformation matrix.
   *
   * @param {Vector2} uv - The uv vector.
   * @return {Vector2} The transformed uv vector.
   */
  transformUv(e) {
    if (this.mapping !== td) return e;
    if (e.applyMatrix3(this.matrix), e.x < 0 || e.x > 1)
      switch (this.wrapS) {
        case El:
          e.x = e.x - Math.floor(e.x);
          break;
        case Hi:
          e.x = e.x < 0 ? 0 : 1;
          break;
        case Tl:
          Math.abs(Math.floor(e.x) % 2) === 1 ? e.x = Math.ceil(e.x) - e.x : e.x = e.x - Math.floor(e.x);
          break;
      }
    if (e.y < 0 || e.y > 1)
      switch (this.wrapT) {
        case El:
          e.y = e.y - Math.floor(e.y);
          break;
        case Hi:
          e.y = e.y < 0 ? 0 : 1;
          break;
        case Tl:
          Math.abs(Math.floor(e.y) % 2) === 1 ? e.y = Math.ceil(e.y) - e.y : e.y = e.y - Math.floor(e.y);
          break;
      }
    return this.flipY && (e.y = 1 - e.y), e;
  }
  /**
   * Setting this property to `true` indicates the engine the texture
   * must be updated in the next render. This triggers a texture upload
   * to the GPU and ensures correct texture parameter configuration.
   *
   * @type {boolean}
   * @default false
   * @param {boolean} value
   */
  set needsUpdate(e) {
    e === !0 && (this.version++, this.source.needsUpdate = !0);
  }
  /**
   * Setting this property to `true` indicates the engine the PMREM
   * must be regenerated.
   *
   * @type {boolean}
   * @default false
   * @param {boolean} value
   */
  set needsPMREMUpdate(e) {
    e === !0 && this.pmremVersion++;
  }
}
$t.DEFAULT_IMAGE = null;
$t.DEFAULT_MAPPING = td;
$t.DEFAULT_ANISOTROPY = 1;
class lt {
  /**
   * Constructs a new 4D vector.
   *
   * @param {number} [x=0] - The x value of this vector.
   * @param {number} [y=0] - The y value of this vector.
   * @param {number} [z=0] - The z value of this vector.
   * @param {number} [w=1] - The w value of this vector.
   */
  constructor(e = 0, t = 0, i = 0, s = 1) {
    lt.prototype.isVector4 = !0, this.x = e, this.y = t, this.z = i, this.w = s;
  }
  /**
   * Alias for {@link Vector4#z}.
   *
   * @type {number}
   */
  get width() {
    return this.z;
  }
  set width(e) {
    this.z = e;
  }
  /**
   * Alias for {@link Vector4#w}.
   *
   * @type {number}
   */
  get height() {
    return this.w;
  }
  set height(e) {
    this.w = e;
  }
  /**
   * Sets the vector components.
   *
   * @param {number} x - The value of the x component.
   * @param {number} y - The value of the y component.
   * @param {number} z - The value of the z component.
   * @param {number} w - The value of the w component.
   * @return {Vector4} A reference to this vector.
   */
  set(e, t, i, s) {
    return this.x = e, this.y = t, this.z = i, this.w = s, this;
  }
  /**
   * Sets the vector components to the same value.
   *
   * @param {number} scalar - The value to set for all vector components.
   * @return {Vector4} A reference to this vector.
   */
  setScalar(e) {
    return this.x = e, this.y = e, this.z = e, this.w = e, this;
  }
  /**
   * Sets the vector's x component to the given value
   *
   * @param {number} x - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setX(e) {
    return this.x = e, this;
  }
  /**
   * Sets the vector's y component to the given value
   *
   * @param {number} y - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setY(e) {
    return this.y = e, this;
  }
  /**
   * Sets the vector's z component to the given value
   *
   * @param {number} z - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setZ(e) {
    return this.z = e, this;
  }
  /**
   * Sets the vector's w component to the given value
   *
   * @param {number} w - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setW(e) {
    return this.w = e, this;
  }
  /**
   * Allows to set a vector component with an index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y,
   * `2` equals to z, `3` equals to w.
   * @param {number} value - The value to set.
   * @return {Vector4} A reference to this vector.
   */
  setComponent(e, t) {
    switch (e) {
      case 0:
        this.x = t;
        break;
      case 1:
        this.y = t;
        break;
      case 2:
        this.z = t;
        break;
      case 3:
        this.w = t;
        break;
      default:
        throw new Error("index is out of range: " + e);
    }
    return this;
  }
  /**
   * Returns the value of the vector component which matches the given index.
   *
   * @param {number} index - The component index. `0` equals to x, `1` equals to y,
   * `2` equals to z, `3` equals to w.
   * @return {number} A vector component value.
   */
  getComponent(e) {
    switch (e) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      case 3:
        return this.w;
      default:
        throw new Error("index is out of range: " + e);
    }
  }
  /**
   * Returns a new vector with copied values from this instance.
   *
   * @return {Vector4} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.x, this.y, this.z, this.w);
  }
  /**
   * Copies the values of the given vector to this instance.
   *
   * @param {Vector3|Vector4} v - The vector to copy.
   * @return {Vector4} A reference to this vector.
   */
  copy(e) {
    return this.x = e.x, this.y = e.y, this.z = e.z, this.w = e.w !== void 0 ? e.w : 1, this;
  }
  /**
   * Adds the given vector to this instance.
   *
   * @param {Vector4} v - The vector to add.
   * @return {Vector4} A reference to this vector.
   */
  add(e) {
    return this.x += e.x, this.y += e.y, this.z += e.z, this.w += e.w, this;
  }
  /**
   * Adds the given scalar value to all components of this instance.
   *
   * @param {number} s - The scalar to add.
   * @return {Vector4} A reference to this vector.
   */
  addScalar(e) {
    return this.x += e, this.y += e, this.z += e, this.w += e, this;
  }
  /**
   * Adds the given vectors and stores the result in this instance.
   *
   * @param {Vector4} a - The first vector.
   * @param {Vector4} b - The second vector.
   * @return {Vector4} A reference to this vector.
   */
  addVectors(e, t) {
    return this.x = e.x + t.x, this.y = e.y + t.y, this.z = e.z + t.z, this.w = e.w + t.w, this;
  }
  /**
   * Adds the given vector scaled by the given factor to this instance.
   *
   * @param {Vector4} v - The vector.
   * @param {number} s - The factor that scales `v`.
   * @return {Vector4} A reference to this vector.
   */
  addScaledVector(e, t) {
    return this.x += e.x * t, this.y += e.y * t, this.z += e.z * t, this.w += e.w * t, this;
  }
  /**
   * Subtracts the given vector from this instance.
   *
   * @param {Vector4} v - The vector to subtract.
   * @return {Vector4} A reference to this vector.
   */
  sub(e) {
    return this.x -= e.x, this.y -= e.y, this.z -= e.z, this.w -= e.w, this;
  }
  /**
   * Subtracts the given scalar value from all components of this instance.
   *
   * @param {number} s - The scalar to subtract.
   * @return {Vector4} A reference to this vector.
   */
  subScalar(e) {
    return this.x -= e, this.y -= e, this.z -= e, this.w -= e, this;
  }
  /**
   * Subtracts the given vectors and stores the result in this instance.
   *
   * @param {Vector4} a - The first vector.
   * @param {Vector4} b - The second vector.
   * @return {Vector4} A reference to this vector.
   */
  subVectors(e, t) {
    return this.x = e.x - t.x, this.y = e.y - t.y, this.z = e.z - t.z, this.w = e.w - t.w, this;
  }
  /**
   * Multiplies the given vector with this instance.
   *
   * @param {Vector4} v - The vector to multiply.
   * @return {Vector4} A reference to this vector.
   */
  multiply(e) {
    return this.x *= e.x, this.y *= e.y, this.z *= e.z, this.w *= e.w, this;
  }
  /**
   * Multiplies the given scalar value with all components of this instance.
   *
   * @param {number} scalar - The scalar to multiply.
   * @return {Vector4} A reference to this vector.
   */
  multiplyScalar(e) {
    return this.x *= e, this.y *= e, this.z *= e, this.w *= e, this;
  }
  /**
   * Multiplies this vector with the given 4x4 matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Vector4} A reference to this vector.
   */
  applyMatrix4(e) {
    const t = this.x, i = this.y, s = this.z, r = this.w, o = e.elements;
    return this.x = o[0] * t + o[4] * i + o[8] * s + o[12] * r, this.y = o[1] * t + o[5] * i + o[9] * s + o[13] * r, this.z = o[2] * t + o[6] * i + o[10] * s + o[14] * r, this.w = o[3] * t + o[7] * i + o[11] * s + o[15] * r, this;
  }
  /**
   * Divides this instance by the given vector.
   *
   * @param {Vector4} v - The vector to divide.
   * @return {Vector4} A reference to this vector.
   */
  divide(e) {
    return this.x /= e.x, this.y /= e.y, this.z /= e.z, this.w /= e.w, this;
  }
  /**
   * Divides this vector by the given scalar.
   *
   * @param {number} scalar - The scalar to divide.
   * @return {Vector4} A reference to this vector.
   */
  divideScalar(e) {
    return this.multiplyScalar(1 / e);
  }
  /**
   * Sets the x, y and z components of this
   * vector to the quaternion's axis and w to the angle.
   *
   * @param {Quaternion} q - The Quaternion to set.
   * @return {Vector4} A reference to this vector.
   */
  setAxisAngleFromQuaternion(e) {
    this.w = 2 * Math.acos(e.w);
    const t = Math.sqrt(1 - e.w * e.w);
    return t < 1e-4 ? (this.x = 1, this.y = 0, this.z = 0) : (this.x = e.x / t, this.y = e.y / t, this.z = e.z / t), this;
  }
  /**
   * Sets the x, y and z components of this
   * vector to the axis of rotation and w to the angle.
   *
   * @param {Matrix4} m - A 4x4 matrix of which the upper left 3x3 matrix is a pure rotation matrix.
   * @return {Vector4} A reference to this vector.
   */
  setAxisAngleFromRotationMatrix(e) {
    let t, i, s, r;
    const l = e.elements, c = l[0], u = l[4], h = l[8], f = l[1], p = l[5], v = l[9], x = l[2], m = l[6], d = l[10];
    if (Math.abs(u - f) < 0.01 && Math.abs(h - x) < 0.01 && Math.abs(v - m) < 0.01) {
      if (Math.abs(u + f) < 0.1 && Math.abs(h + x) < 0.1 && Math.abs(v + m) < 0.1 && Math.abs(c + p + d - 3) < 0.1)
        return this.set(1, 0, 0, 0), this;
      t = Math.PI;
      const A = (c + 1) / 2, M = (p + 1) / 2, R = (d + 1) / 2, w = (u + f) / 4, D = (h + x) / 4, U = (v + m) / 4;
      return A > M && A > R ? A < 0.01 ? (i = 0, s = 0.707106781, r = 0.707106781) : (i = Math.sqrt(A), s = w / i, r = D / i) : M > R ? M < 0.01 ? (i = 0.707106781, s = 0, r = 0.707106781) : (s = Math.sqrt(M), i = w / s, r = U / s) : R < 0.01 ? (i = 0.707106781, s = 0.707106781, r = 0) : (r = Math.sqrt(R), i = D / r, s = U / r), this.set(i, s, r, t), this;
    }
    let b = Math.sqrt((m - v) * (m - v) + (h - x) * (h - x) + (f - u) * (f - u));
    return Math.abs(b) < 1e-3 && (b = 1), this.x = (m - v) / b, this.y = (h - x) / b, this.z = (f - u) / b, this.w = Math.acos((c + p + d - 1) / 2), this;
  }
  /**
   * Sets the vector components to the position elements of the
   * given transformation matrix.
   *
   * @param {Matrix4} m - The 4x4 matrix.
   * @return {Vector4} A reference to this vector.
   */
  setFromMatrixPosition(e) {
    const t = e.elements;
    return this.x = t[12], this.y = t[13], this.z = t[14], this.w = t[15], this;
  }
  /**
   * If this vector's x, y, z or w value is greater than the given vector's x, y, z or w
   * value, replace that value with the corresponding min value.
   *
   * @param {Vector4} v - The vector.
   * @return {Vector4} A reference to this vector.
   */
  min(e) {
    return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this.z = Math.min(this.z, e.z), this.w = Math.min(this.w, e.w), this;
  }
  /**
   * If this vector's x, y, z or w value is less than the given vector's x, y, z or w
   * value, replace that value with the corresponding max value.
   *
   * @param {Vector4} v - The vector.
   * @return {Vector4} A reference to this vector.
   */
  max(e) {
    return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this.z = Math.max(this.z, e.z), this.w = Math.max(this.w, e.w), this;
  }
  /**
   * If this vector's x, y, z or w value is greater than the max vector's x, y, z or w
   * value, it is replaced by the corresponding value.
   * If this vector's x, y, z or w value is less than the min vector's x, y, z or w value,
   * it is replaced by the corresponding value.
   *
   * @param {Vector4} min - The minimum x, y and z values.
   * @param {Vector4} max - The maximum x, y and z values in the desired range.
   * @return {Vector4} A reference to this vector.
   */
  clamp(e, t) {
    return this.x = je(this.x, e.x, t.x), this.y = je(this.y, e.y, t.y), this.z = je(this.z, e.z, t.z), this.w = je(this.w, e.w, t.w), this;
  }
  /**
   * If this vector's x, y, z or w values are greater than the max value, they are
   * replaced by the max value.
   * If this vector's x, y, z or w values are less than the min value, they are
   * replaced by the min value.
   *
   * @param {number} minVal - The minimum value the components will be clamped to.
   * @param {number} maxVal - The maximum value the components will be clamped to.
   * @return {Vector4} A reference to this vector.
   */
  clampScalar(e, t) {
    return this.x = je(this.x, e, t), this.y = je(this.y, e, t), this.z = je(this.z, e, t), this.w = je(this.w, e, t), this;
  }
  /**
   * If this vector's length is greater than the max value, it is replaced by
   * the max value.
   * If this vector's length is less than the min value, it is replaced by the
   * min value.
   *
   * @param {number} min - The minimum value the vector length will be clamped to.
   * @param {number} max - The maximum value the vector length will be clamped to.
   * @return {Vector4} A reference to this vector.
   */
  clampLength(e, t) {
    const i = this.length();
    return this.divideScalar(i || 1).multiplyScalar(je(i, e, t));
  }
  /**
   * The components of this vector are rounded down to the nearest integer value.
   *
   * @return {Vector4} A reference to this vector.
   */
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this.w = Math.floor(this.w), this;
  }
  /**
   * The components of this vector are rounded up to the nearest integer value.
   *
   * @return {Vector4} A reference to this vector.
   */
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this.w = Math.ceil(this.w), this;
  }
  /**
   * The components of this vector are rounded to the nearest integer value
   *
   * @return {Vector4} A reference to this vector.
   */
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this.w = Math.round(this.w), this;
  }
  /**
   * The components of this vector are rounded towards zero (up if negative,
   * down if positive) to an integer value.
   *
   * @return {Vector4} A reference to this vector.
   */
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this.w = Math.trunc(this.w), this;
  }
  /**
   * Inverts this vector - i.e. sets x = -x, y = -y, z = -z, w = -w.
   *
   * @return {Vector4} A reference to this vector.
   */
  negate() {
    return this.x = -this.x, this.y = -this.y, this.z = -this.z, this.w = -this.w, this;
  }
  /**
   * Calculates the dot product of the given vector with this instance.
   *
   * @param {Vector4} v - The vector to compute the dot product with.
   * @return {number} The result of the dot product.
   */
  dot(e) {
    return this.x * e.x + this.y * e.y + this.z * e.z + this.w * e.w;
  }
  /**
   * Computes the square of the Euclidean length (straight-line length) from
   * (0, 0, 0, 0) to (x, y, z, w). If you are comparing the lengths of vectors, you should
   * compare the length squared instead as it is slightly more efficient to calculate.
   *
   * @return {number} The square length of this vector.
   */
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
  }
  /**
   * Computes the  Euclidean length (straight-line length) from (0, 0, 0, 0) to (x, y, z, w).
   *
   * @return {number} The length of this vector.
   */
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
  }
  /**
   * Computes the Manhattan length of this vector.
   *
   * @return {number} The length of this vector.
   */
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z) + Math.abs(this.w);
  }
  /**
   * Converts this vector to a unit vector - that is, sets it equal to a vector
   * with the same direction as this one, but with a vector length of `1`.
   *
   * @return {Vector4} A reference to this vector.
   */
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  /**
   * Sets this vector to a vector with the same direction as this one, but
   * with the specified length.
   *
   * @param {number} length - The new length of this vector.
   * @return {Vector4} A reference to this vector.
   */
  setLength(e) {
    return this.normalize().multiplyScalar(e);
  }
  /**
   * Linearly interpolates between the given vector and this instance, where
   * alpha is the percent distance along the line - alpha = 0 will be this
   * vector, and alpha = 1 will be the given one.
   *
   * @param {Vector4} v - The vector to interpolate towards.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector4} A reference to this vector.
   */
  lerp(e, t) {
    return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this.z += (e.z - this.z) * t, this.w += (e.w - this.w) * t, this;
  }
  /**
   * Linearly interpolates between the given vectors, where alpha is the percent
   * distance along the line - alpha = 0 will be first vector, and alpha = 1 will
   * be the second one. The result is stored in this instance.
   *
   * @param {Vector4} v1 - The first vector.
   * @param {Vector4} v2 - The second vector.
   * @param {number} alpha - The interpolation factor, typically in the closed interval `[0, 1]`.
   * @return {Vector4} A reference to this vector.
   */
  lerpVectors(e, t, i) {
    return this.x = e.x + (t.x - e.x) * i, this.y = e.y + (t.y - e.y) * i, this.z = e.z + (t.z - e.z) * i, this.w = e.w + (t.w - e.w) * i, this;
  }
  /**
   * Returns `true` if this vector is equal with the given one.
   *
   * @param {Vector4} v - The vector to test for equality.
   * @return {boolean} Whether this vector is equal with the given one.
   */
  equals(e) {
    return e.x === this.x && e.y === this.y && e.z === this.z && e.w === this.w;
  }
  /**
   * Sets this vector's x value to be `array[ offset ]`, y value to be `array[ offset + 1 ]`,
   * z value to be `array[ offset + 2 ]`, w value to be `array[ offset + 3 ]`.
   *
   * @param {Array<number>} array - An array holding the vector component values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Vector4} A reference to this vector.
   */
  fromArray(e, t = 0) {
    return this.x = e[t], this.y = e[t + 1], this.z = e[t + 2], this.w = e[t + 3], this;
  }
  /**
   * Writes the components of this vector to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the vector components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The vector components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this.x, e[t + 1] = this.y, e[t + 2] = this.z, e[t + 3] = this.w, e;
  }
  /**
   * Sets the components of this vector from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding vector data.
   * @param {number} index - The index into the attribute.
   * @return {Vector4} A reference to this vector.
   */
  fromBufferAttribute(e, t) {
    return this.x = e.getX(t), this.y = e.getY(t), this.z = e.getZ(t), this.w = e.getW(t), this;
  }
  /**
   * Sets each component of this vector to a pseudo-random value between `0` and
   * `1`, excluding `1`.
   *
   * @return {Vector4} A reference to this vector.
   */
  random() {
    return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this.w = Math.random(), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y, yield this.z, yield this.w;
  }
}
class Ng extends Zi {
  /**
   * Render target options.
   *
   * @typedef {Object} RenderTarget~Options
   * @property {boolean} [generateMipmaps=false] - Whether to generate mipmaps or not.
   * @property {number} [magFilter=LinearFilter] - The mag filter.
   * @property {number} [minFilter=LinearFilter] - The min filter.
   * @property {number} [format=RGBAFormat] - The texture format.
   * @property {number} [type=UnsignedByteType] - The texture type.
   * @property {?string} [internalFormat=null] - The texture's internal format.
   * @property {number} [wrapS=ClampToEdgeWrapping] - The texture's uv wrapping mode.
   * @property {number} [wrapT=ClampToEdgeWrapping] - The texture's uv wrapping mode.
   * @property {number} [anisotropy=1] - The texture's anisotropy value.
   * @property {string} [colorSpace=NoColorSpace] - The texture's color space.
   * @property {boolean} [depthBuffer=true] - Whether to allocate a depth buffer or not.
   * @property {boolean} [stencilBuffer=false] - Whether to allocate a stencil buffer or not.
   * @property {boolean} [resolveDepthBuffer=true] - Whether to resolve the depth buffer or not.
   * @property {boolean} [resolveStencilBuffer=true] - Whether  to resolve the stencil buffer or not.
   * @property {?Texture} [depthTexture=null] - Reference to a depth texture.
   * @property {number} [samples=0] - The MSAA samples count.
   * @property {number} [count=1] - Defines the number of color attachments . Must be at least `1`.
   * @property {number} [depth=1] - The texture depth.
   * @property {boolean} [multiview=false] - Whether this target is used for multiview rendering.
   */
  /**
   * Constructs a new render target.
   *
   * @param {number} [width=1] - The width of the render target.
   * @param {number} [height=1] - The height of the render target.
   * @param {RenderTarget~Options} [options] - The configuration object.
   */
  constructor(e = 1, t = 1, i = {}) {
    super(), i = Object.assign({
      generateMipmaps: !1,
      internalFormat: null,
      minFilter: Un,
      depthBuffer: !0,
      stencilBuffer: !1,
      resolveDepthBuffer: !0,
      resolveStencilBuffer: !0,
      depthTexture: null,
      samples: 0,
      count: 1,
      depth: 1,
      multiview: !1
    }, i), this.isRenderTarget = !0, this.width = e, this.height = t, this.depth = i.depth, this.scissor = new lt(0, 0, e, t), this.scissorTest = !1, this.viewport = new lt(0, 0, e, t);
    const s = { width: e, height: t, depth: i.depth }, r = new $t(s);
    this.textures = [];
    const o = i.count;
    for (let a = 0; a < o; a++)
      this.textures[a] = r.clone(), this.textures[a].isRenderTargetTexture = !0, this.textures[a].renderTarget = this;
    this._setTextureOptions(i), this.depthBuffer = i.depthBuffer, this.stencilBuffer = i.stencilBuffer, this.resolveDepthBuffer = i.resolveDepthBuffer, this.resolveStencilBuffer = i.resolveStencilBuffer, this._depthTexture = null, this.depthTexture = i.depthTexture, this.samples = i.samples, this.multiview = i.multiview;
  }
  _setTextureOptions(e = {}) {
    const t = {
      minFilter: Un,
      generateMipmaps: !1,
      flipY: !1,
      internalFormat: null
    };
    e.mapping !== void 0 && (t.mapping = e.mapping), e.wrapS !== void 0 && (t.wrapS = e.wrapS), e.wrapT !== void 0 && (t.wrapT = e.wrapT), e.wrapR !== void 0 && (t.wrapR = e.wrapR), e.magFilter !== void 0 && (t.magFilter = e.magFilter), e.minFilter !== void 0 && (t.minFilter = e.minFilter), e.format !== void 0 && (t.format = e.format), e.type !== void 0 && (t.type = e.type), e.anisotropy !== void 0 && (t.anisotropy = e.anisotropy), e.colorSpace !== void 0 && (t.colorSpace = e.colorSpace), e.flipY !== void 0 && (t.flipY = e.flipY), e.generateMipmaps !== void 0 && (t.generateMipmaps = e.generateMipmaps), e.internalFormat !== void 0 && (t.internalFormat = e.internalFormat);
    for (let i = 0; i < this.textures.length; i++)
      this.textures[i].setValues(t);
  }
  /**
   * The texture representing the default color attachment.
   *
   * @type {Texture}
   */
  get texture() {
    return this.textures[0];
  }
  set texture(e) {
    this.textures[0] = e;
  }
  set depthTexture(e) {
    this._depthTexture !== null && (this._depthTexture.renderTarget = null), e !== null && (e.renderTarget = this), this._depthTexture = e;
  }
  /**
   * Instead of saving the depth in a renderbuffer, a texture
   * can be used instead which is useful for further processing
   * e.g. in context of post-processing.
   *
   * @type {?DepthTexture}
   * @default null
   */
  get depthTexture() {
    return this._depthTexture;
  }
  /**
   * Sets the size of this render target.
   *
   * @param {number} width - The width.
   * @param {number} height - The height.
   * @param {number} [depth=1] - The depth.
   */
  setSize(e, t, i = 1) {
    if (this.width !== e || this.height !== t || this.depth !== i) {
      this.width = e, this.height = t, this.depth = i;
      for (let s = 0, r = this.textures.length; s < r; s++)
        this.textures[s].image.width = e, this.textures[s].image.height = t, this.textures[s].image.depth = i, this.textures[s].isArrayTexture = this.textures[s].image.depth > 1;
      this.dispose();
    }
    this.viewport.set(0, 0, e, t), this.scissor.set(0, 0, e, t);
  }
  /**
   * Returns a new render target with copied values from this instance.
   *
   * @return {RenderTarget} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the settings of the given render target. This is a structural copy so
   * no resources are shared between render targets after the copy. That includes
   * all MRT textures and the depth texture.
   *
   * @param {RenderTarget} source - The render target to copy.
   * @return {RenderTarget} A reference to this instance.
   */
  copy(e) {
    this.width = e.width, this.height = e.height, this.depth = e.depth, this.scissor.copy(e.scissor), this.scissorTest = e.scissorTest, this.viewport.copy(e.viewport), this.textures.length = 0;
    for (let t = 0, i = e.textures.length; t < i; t++) {
      this.textures[t] = e.textures[t].clone(), this.textures[t].isRenderTargetTexture = !0, this.textures[t].renderTarget = this;
      const s = Object.assign({}, e.textures[t].image);
      this.textures[t].source = new bc(s);
    }
    return this.depthBuffer = e.depthBuffer, this.stencilBuffer = e.stencilBuffer, this.resolveDepthBuffer = e.resolveDepthBuffer, this.resolveStencilBuffer = e.resolveStencilBuffer, e.depthTexture !== null && (this.depthTexture = e.depthTexture.clone()), this.samples = e.samples, this;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   *
   * @fires RenderTarget#dispose
   */
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
}
class qi extends Ng {
  /**
   * Constructs a new 3D render target.
   *
   * @param {number} [width=1] - The width of the render target.
   * @param {number} [height=1] - The height of the render target.
   * @param {RenderTarget~Options} [options] - The configuration object.
   */
  constructor(e = 1, t = 1, i = {}) {
    super(e, t, i), this.isWebGLRenderTarget = !0;
  }
}
class dd extends $t {
  /**
   * Constructs a new data array texture.
   *
   * @param {?TypedArray} [data=null] - The buffer data.
   * @param {number} [width=1] - The width of the texture.
   * @param {number} [height=1] - The height of the texture.
   * @param {number} [depth=1] - The depth of the texture.
   */
  constructor(e = null, t = 1, i = 1, s = 1) {
    super(null), this.isDataArrayTexture = !0, this.image = { data: e, width: t, height: i, depth: s }, this.magFilter = yn, this.minFilter = yn, this.wrapR = Hi, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1, this.layerUpdates = /* @__PURE__ */ new Set();
  }
  /**
   * Describes that a specific layer of the texture needs to be updated.
   * Normally when {@link Texture#needsUpdate} is set to `true`, the
   * entire data texture array is sent to the GPU. Marking specific
   * layers will only transmit subsets of all mipmaps associated with a
   * specific depth in the array which is often much more performant.
   *
   * @param {number} layerIndex - The layer index that should be updated.
   */
  addLayerUpdate(e) {
    this.layerUpdates.add(e);
  }
  /**
   * Resets the layer updates registry.
   */
  clearLayerUpdates() {
    this.layerUpdates.clear();
  }
}
class Fg extends $t {
  /**
   * Constructs a new data array texture.
   *
   * @param {?TypedArray} [data=null] - The buffer data.
   * @param {number} [width=1] - The width of the texture.
   * @param {number} [height=1] - The height of the texture.
   * @param {number} [depth=1] - The depth of the texture.
   */
  constructor(e = null, t = 1, i = 1, s = 1) {
    super(null), this.isData3DTexture = !0, this.image = { data: e, width: t, height: i, depth: s }, this.magFilter = yn, this.minFilter = yn, this.wrapR = Hi, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1;
  }
}
class Ur {
  /**
   * Constructs a new bounding box.
   *
   * @param {Vector3} [min=(Infinity,Infinity,Infinity)] - A vector representing the lower boundary of the box.
   * @param {Vector3} [max=(-Infinity,-Infinity,-Infinity)] - A vector representing the upper boundary of the box.
   */
  constructor(e = new N(1 / 0, 1 / 0, 1 / 0), t = new N(-1 / 0, -1 / 0, -1 / 0)) {
    this.isBox3 = !0, this.min = e, this.max = t;
  }
  /**
   * Sets the lower and upper boundaries of this box.
   * Please note that this method only copies the values from the given objects.
   *
   * @param {Vector3} min - The lower boundary of the box.
   * @param {Vector3} max - The upper boundary of the box.
   * @return {Box3} A reference to this bounding box.
   */
  set(e, t) {
    return this.min.copy(e), this.max.copy(t), this;
  }
  /**
   * Sets the upper and lower bounds of this box so it encloses the position data
   * in the given array.
   *
   * @param {Array<number>} array - An array holding 3D position data.
   * @return {Box3} A reference to this bounding box.
   */
  setFromArray(e) {
    this.makeEmpty();
    for (let t = 0, i = e.length; t < i; t += 3)
      this.expandByPoint(_n.fromArray(e, t));
    return this;
  }
  /**
   * Sets the upper and lower bounds of this box so it encloses the position data
   * in the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - A buffer attribute holding 3D position data.
   * @return {Box3} A reference to this bounding box.
   */
  setFromBufferAttribute(e) {
    this.makeEmpty();
    for (let t = 0, i = e.count; t < i; t++)
      this.expandByPoint(_n.fromBufferAttribute(e, t));
    return this;
  }
  /**
   * Sets the upper and lower bounds of this box so it encloses the position data
   * in the given array.
   *
   * @param {Array<Vector3>} points - An array holding 3D position data as instances of {@link Vector3}.
   * @return {Box3} A reference to this bounding box.
   */
  setFromPoints(e) {
    this.makeEmpty();
    for (let t = 0, i = e.length; t < i; t++)
      this.expandByPoint(e[t]);
    return this;
  }
  /**
   * Centers this box on the given center vector and sets this box's width, height and
   * depth to the given size values.
   *
   * @param {Vector3} center - The center of the box.
   * @param {Vector3} size - The x, y and z dimensions of the box.
   * @return {Box3} A reference to this bounding box.
   */
  setFromCenterAndSize(e, t) {
    const i = _n.copy(t).multiplyScalar(0.5);
    return this.min.copy(e).sub(i), this.max.copy(e).add(i), this;
  }
  /**
   * Computes the world-axis-aligned bounding box for the given 3D object
   * (including its children), accounting for the object's, and children's,
   * world transforms. The function may result in a larger box than strictly necessary.
   *
   * @param {Object3D} object - The 3D object to compute the bounding box for.
   * @param {boolean} [precise=false] - If set to `true`, the method computes the smallest
   * world-axis-aligned bounding box at the expense of more computation.
   * @return {Box3} A reference to this bounding box.
   */
  setFromObject(e, t = !1) {
    return this.makeEmpty(), this.expandByObject(e, t);
  }
  /**
   * Returns a new box with copied values from this instance.
   *
   * @return {Box3} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given box to this instance.
   *
   * @param {Box3} box - The box to copy.
   * @return {Box3} A reference to this bounding box.
   */
  copy(e) {
    return this.min.copy(e.min), this.max.copy(e.max), this;
  }
  /**
   * Makes this box empty which means in encloses a zero space in 3D.
   *
   * @return {Box3} A reference to this bounding box.
   */
  makeEmpty() {
    return this.min.x = this.min.y = this.min.z = 1 / 0, this.max.x = this.max.y = this.max.z = -1 / 0, this;
  }
  /**
   * Returns true if this box includes zero points within its bounds.
   * Note that a box with equal lower and upper bounds still includes one
   * point, the one both bounds share.
   *
   * @return {boolean} Whether this box is empty or not.
   */
  isEmpty() {
    return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z;
  }
  /**
   * Returns the center point of this box.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The center point.
   */
  getCenter(e) {
    return this.isEmpty() ? e.set(0, 0, 0) : e.addVectors(this.min, this.max).multiplyScalar(0.5);
  }
  /**
   * Returns the dimensions of this box.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The size.
   */
  getSize(e) {
    return this.isEmpty() ? e.set(0, 0, 0) : e.subVectors(this.max, this.min);
  }
  /**
   * Expands the boundaries of this box to include the given point.
   *
   * @param {Vector3} point - The point that should be included by the bounding box.
   * @return {Box3} A reference to this bounding box.
   */
  expandByPoint(e) {
    return this.min.min(e), this.max.max(e), this;
  }
  /**
   * Expands this box equilaterally by the given vector. The width of this
   * box will be expanded by the x component of the vector in both
   * directions. The height of this box will be expanded by the y component of
   * the vector in both directions. The depth of this box will be
   * expanded by the z component of the vector in both directions.
   *
   * @param {Vector3} vector - The vector that should expand the bounding box.
   * @return {Box3} A reference to this bounding box.
   */
  expandByVector(e) {
    return this.min.sub(e), this.max.add(e), this;
  }
  /**
   * Expands each dimension of the box by the given scalar. If negative, the
   * dimensions of the box will be contracted.
   *
   * @param {number} scalar - The scalar value that should expand the bounding box.
   * @return {Box3} A reference to this bounding box.
   */
  expandByScalar(e) {
    return this.min.addScalar(-e), this.max.addScalar(e), this;
  }
  /**
   * Expands the boundaries of this box to include the given 3D object and
   * its children, accounting for the object's, and children's, world
   * transforms. The function may result in a larger box than strictly
   * necessary (unless the precise parameter is set to true).
   *
   * @param {Object3D} object - The 3D object that should expand the bounding box.
   * @param {boolean} precise - If set to `true`, the method expands the bounding box
   * as little as necessary at the expense of more computation.
   * @return {Box3} A reference to this bounding box.
   */
  expandByObject(e, t = !1) {
    e.updateWorldMatrix(!1, !1);
    const i = e.geometry;
    if (i !== void 0) {
      const r = i.getAttribute("position");
      if (t === !0 && r !== void 0 && e.isInstancedMesh !== !0)
        for (let o = 0, a = r.count; o < a; o++)
          e.isMesh === !0 ? e.getVertexPosition(o, _n) : _n.fromBufferAttribute(r, o), _n.applyMatrix4(e.matrixWorld), this.expandByPoint(_n);
      else
        e.boundingBox !== void 0 ? (e.boundingBox === null && e.computeBoundingBox(), kr.copy(e.boundingBox)) : (i.boundingBox === null && i.computeBoundingBox(), kr.copy(i.boundingBox)), kr.applyMatrix4(e.matrixWorld), this.union(kr);
    }
    const s = e.children;
    for (let r = 0, o = s.length; r < o; r++)
      this.expandByObject(s[r], t);
    return this;
  }
  /**
   * Returns `true` if the given point lies within or on the boundaries of this box.
   *
   * @param {Vector3} point - The point to test.
   * @return {boolean} Whether the bounding box contains the given point or not.
   */
  containsPoint(e) {
    return e.x >= this.min.x && e.x <= this.max.x && e.y >= this.min.y && e.y <= this.max.y && e.z >= this.min.z && e.z <= this.max.z;
  }
  /**
   * Returns `true` if this bounding box includes the entirety of the given bounding box.
   * If this box and the given one are identical, this function also returns `true`.
   *
   * @param {Box3} box - The bounding box to test.
   * @return {boolean} Whether the bounding box contains the given bounding box or not.
   */
  containsBox(e) {
    return this.min.x <= e.min.x && e.max.x <= this.max.x && this.min.y <= e.min.y && e.max.y <= this.max.y && this.min.z <= e.min.z && e.max.z <= this.max.z;
  }
  /**
   * Returns a point as a proportion of this box's width, height and depth.
   *
   * @param {Vector3} point - A point in 3D space.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} A point as a proportion of this box's width, height and depth.
   */
  getParameter(e, t) {
    return t.set(
      (e.x - this.min.x) / (this.max.x - this.min.x),
      (e.y - this.min.y) / (this.max.y - this.min.y),
      (e.z - this.min.z) / (this.max.z - this.min.z)
    );
  }
  /**
   * Returns `true` if the given bounding box intersects with this bounding box.
   *
   * @param {Box3} box - The bounding box to test.
   * @return {boolean} Whether the given bounding box intersects with this bounding box.
   */
  intersectsBox(e) {
    return e.max.x >= this.min.x && e.min.x <= this.max.x && e.max.y >= this.min.y && e.min.y <= this.max.y && e.max.z >= this.min.z && e.min.z <= this.max.z;
  }
  /**
   * Returns `true` if the given bounding sphere intersects with this bounding box.
   *
   * @param {Sphere} sphere - The bounding sphere to test.
   * @return {boolean} Whether the given bounding sphere intersects with this bounding box.
   */
  intersectsSphere(e) {
    return this.clampPoint(e.center, _n), _n.distanceToSquared(e.center) <= e.radius * e.radius;
  }
  /**
   * Returns `true` if the given plane intersects with this bounding box.
   *
   * @param {Plane} plane - The plane to test.
   * @return {boolean} Whether the given plane intersects with this bounding box.
   */
  intersectsPlane(e) {
    let t, i;
    return e.normal.x > 0 ? (t = e.normal.x * this.min.x, i = e.normal.x * this.max.x) : (t = e.normal.x * this.max.x, i = e.normal.x * this.min.x), e.normal.y > 0 ? (t += e.normal.y * this.min.y, i += e.normal.y * this.max.y) : (t += e.normal.y * this.max.y, i += e.normal.y * this.min.y), e.normal.z > 0 ? (t += e.normal.z * this.min.z, i += e.normal.z * this.max.z) : (t += e.normal.z * this.max.z, i += e.normal.z * this.min.z), t <= -e.constant && i >= -e.constant;
  }
  /**
   * Returns `true` if the given triangle intersects with this bounding box.
   *
   * @param {Triangle} triangle - The triangle to test.
   * @return {boolean} Whether the given triangle intersects with this bounding box.
   */
  intersectsTriangle(e) {
    if (this.isEmpty())
      return !1;
    this.getCenter($s), Gr.subVectors(this.max, $s), ss.subVectors(e.a, $s), rs.subVectors(e.b, $s), os.subVectors(e.c, $s), ai.subVectors(rs, ss), li.subVectors(os, rs), Ci.subVectors(ss, os);
    let t = [
      0,
      -ai.z,
      ai.y,
      0,
      -li.z,
      li.y,
      0,
      -Ci.z,
      Ci.y,
      ai.z,
      0,
      -ai.x,
      li.z,
      0,
      -li.x,
      Ci.z,
      0,
      -Ci.x,
      -ai.y,
      ai.x,
      0,
      -li.y,
      li.x,
      0,
      -Ci.y,
      Ci.x,
      0
    ];
    return !ba(t, ss, rs, os, Gr) || (t = [1, 0, 0, 0, 1, 0, 0, 0, 1], !ba(t, ss, rs, os, Gr)) ? !1 : (Wr.crossVectors(ai, li), t = [Wr.x, Wr.y, Wr.z], ba(t, ss, rs, os, Gr));
  }
  /**
   * Clamps the given point within the bounds of this box.
   *
   * @param {Vector3} point - The point to clamp.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The clamped point.
   */
  clampPoint(e, t) {
    return t.copy(e).clamp(this.min, this.max);
  }
  /**
   * Returns the euclidean distance from any edge of this box to the specified point. If
   * the given point lies inside of this box, the distance will be `0`.
   *
   * @param {Vector3} point - The point to compute the distance to.
   * @return {number} The euclidean distance.
   */
  distanceToPoint(e) {
    return this.clampPoint(e, _n).distanceTo(e);
  }
  /**
   * Returns a bounding sphere that encloses this bounding box.
   *
   * @param {Sphere} target - The target sphere that is used to store the method's result.
   * @return {Sphere} The bounding sphere that encloses this bounding box.
   */
  getBoundingSphere(e) {
    return this.isEmpty() ? e.makeEmpty() : (this.getCenter(e.center), e.radius = this.getSize(_n).length() * 0.5), e;
  }
  /**
   * Computes the intersection of this bounding box and the given one, setting the upper
   * bound of this box to the lesser of the two boxes' upper bounds and the
   * lower bound of this box to the greater of the two boxes' lower bounds. If
   * there's no overlap, makes this box empty.
   *
   * @param {Box3} box - The bounding box to intersect with.
   * @return {Box3} A reference to this bounding box.
   */
  intersect(e) {
    return this.min.max(e.min), this.max.min(e.max), this.isEmpty() && this.makeEmpty(), this;
  }
  /**
   * Computes the union of this box and another and the given one, setting the upper
   * bound of this box to the greater of the two boxes' upper bounds and the
   * lower bound of this box to the lesser of the two boxes' lower bounds.
   *
   * @param {Box3} box - The bounding box that will be unioned with this instance.
   * @return {Box3} A reference to this bounding box.
   */
  union(e) {
    return this.min.min(e.min), this.max.max(e.max), this;
  }
  /**
   * Transforms this bounding box by the given 4x4 transformation matrix.
   *
   * @param {Matrix4} matrix - The transformation matrix.
   * @return {Box3} A reference to this bounding box.
   */
  applyMatrix4(e) {
    return this.isEmpty() ? this : (Gn[0].set(this.min.x, this.min.y, this.min.z).applyMatrix4(e), Gn[1].set(this.min.x, this.min.y, this.max.z).applyMatrix4(e), Gn[2].set(this.min.x, this.max.y, this.min.z).applyMatrix4(e), Gn[3].set(this.min.x, this.max.y, this.max.z).applyMatrix4(e), Gn[4].set(this.max.x, this.min.y, this.min.z).applyMatrix4(e), Gn[5].set(this.max.x, this.min.y, this.max.z).applyMatrix4(e), Gn[6].set(this.max.x, this.max.y, this.min.z).applyMatrix4(e), Gn[7].set(this.max.x, this.max.y, this.max.z).applyMatrix4(e), this.setFromPoints(Gn), this);
  }
  /**
   * Adds the given offset to both the upper and lower bounds of this bounding box,
   * effectively moving it in 3D space.
   *
   * @param {Vector3} offset - The offset that should be used to translate the bounding box.
   * @return {Box3} A reference to this bounding box.
   */
  translate(e) {
    return this.min.add(e), this.max.add(e), this;
  }
  /**
   * Returns `true` if this bounding box is equal with the given one.
   *
   * @param {Box3} box - The box to test for equality.
   * @return {boolean} Whether this bounding box is equal with the given one.
   */
  equals(e) {
    return e.min.equals(this.min) && e.max.equals(this.max);
  }
  /**
   * Returns a serialized structure of the bounding box.
   *
   * @return {Object} Serialized structure with fields representing the object state.
   */
  toJSON() {
    return {
      min: this.min.toArray(),
      max: this.max.toArray()
    };
  }
  /**
   * Returns a serialized structure of the bounding box.
   *
   * @param {Object} json - The serialized json to set the box from.
   * @return {Box3} A reference to this bounding box.
   */
  fromJSON(e) {
    return this.min.fromArray(e.min), this.max.fromArray(e.max), this;
  }
}
const Gn = [
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N(),
  /* @__PURE__ */ new N()
], _n = /* @__PURE__ */ new N(), kr = /* @__PURE__ */ new Ur(), ss = /* @__PURE__ */ new N(), rs = /* @__PURE__ */ new N(), os = /* @__PURE__ */ new N(), ai = /* @__PURE__ */ new N(), li = /* @__PURE__ */ new N(), Ci = /* @__PURE__ */ new N(), $s = /* @__PURE__ */ new N(), Gr = /* @__PURE__ */ new N(), Wr = /* @__PURE__ */ new N(), Pi = /* @__PURE__ */ new N();
function ba(n, e, t, i, s) {
  for (let r = 0, o = n.length - 3; r <= o; r += 3) {
    Pi.fromArray(n, r);
    const a = s.x * Math.abs(Pi.x) + s.y * Math.abs(Pi.y) + s.z * Math.abs(Pi.z), l = e.dot(Pi), c = t.dot(Pi), u = i.dot(Pi);
    if (Math.max(-Math.max(l, c, u), Math.min(l, c, u)) > a)
      return !1;
  }
  return !0;
}
const Og = /* @__PURE__ */ new Ur(), Zs = /* @__PURE__ */ new N(), Aa = /* @__PURE__ */ new N();
class Nr {
  /**
   * Constructs a new sphere.
   *
   * @param {Vector3} [center=(0,0,0)] - The center of the sphere
   * @param {number} [radius=-1] - The radius of the sphere.
   */
  constructor(e = new N(), t = -1) {
    this.isSphere = !0, this.center = e, this.radius = t;
  }
  /**
   * Sets the sphere's components by copying the given values.
   *
   * @param {Vector3} center - The center.
   * @param {number} radius - The radius.
   * @return {Sphere} A reference to this sphere.
   */
  set(e, t) {
    return this.center.copy(e), this.radius = t, this;
  }
  /**
   * Computes the minimum bounding sphere for list of points.
   * If the optional center point is given, it is used as the sphere's
   * center. Otherwise, the center of the axis-aligned bounding box
   * encompassing the points is calculated.
   *
   * @param {Array<Vector3>} points - A list of points in 3D space.
   * @param {Vector3} [optionalCenter] - The center of the sphere.
   * @return {Sphere} A reference to this sphere.
   */
  setFromPoints(e, t) {
    const i = this.center;
    t !== void 0 ? i.copy(t) : Og.setFromPoints(e).getCenter(i);
    let s = 0;
    for (let r = 0, o = e.length; r < o; r++)
      s = Math.max(s, i.distanceToSquared(e[r]));
    return this.radius = Math.sqrt(s), this;
  }
  /**
   * Copies the values of the given sphere to this instance.
   *
   * @param {Sphere} sphere - The sphere to copy.
   * @return {Sphere} A reference to this sphere.
   */
  copy(e) {
    return this.center.copy(e.center), this.radius = e.radius, this;
  }
  /**
   * Returns `true` if the sphere is empty (the radius set to a negative number).
   *
   * Spheres with a radius of `0` contain only their center point and are not
   * considered to be empty.
   *
   * @return {boolean} Whether this sphere is empty or not.
   */
  isEmpty() {
    return this.radius < 0;
  }
  /**
   * Makes this sphere empty which means in encloses a zero space in 3D.
   *
   * @return {Sphere} A reference to this sphere.
   */
  makeEmpty() {
    return this.center.set(0, 0, 0), this.radius = -1, this;
  }
  /**
   * Returns `true` if this sphere contains the given point inclusive of
   * the surface of the sphere.
   *
   * @param {Vector3} point - The point to check.
   * @return {boolean} Whether this sphere contains the given point or not.
   */
  containsPoint(e) {
    return e.distanceToSquared(this.center) <= this.radius * this.radius;
  }
  /**
   * Returns the closest distance from the boundary of the sphere to the
   * given point. If the sphere contains the point, the distance will
   * be negative.
   *
   * @param {Vector3} point - The point to compute the distance to.
   * @return {number} The distance to the point.
   */
  distanceToPoint(e) {
    return e.distanceTo(this.center) - this.radius;
  }
  /**
   * Returns `true` if this sphere intersects with the given one.
   *
   * @param {Sphere} sphere - The sphere to test.
   * @return {boolean} Whether this sphere intersects with the given one or not.
   */
  intersectsSphere(e) {
    const t = this.radius + e.radius;
    return e.center.distanceToSquared(this.center) <= t * t;
  }
  /**
   * Returns `true` if this sphere intersects with the given box.
   *
   * @param {Box3} box - The box to test.
   * @return {boolean} Whether this sphere intersects with the given box or not.
   */
  intersectsBox(e) {
    return e.intersectsSphere(this);
  }
  /**
   * Returns `true` if this sphere intersects with the given plane.
   *
   * @param {Plane} plane - The plane to test.
   * @return {boolean} Whether this sphere intersects with the given plane or not.
   */
  intersectsPlane(e) {
    return Math.abs(e.distanceToPoint(this.center)) <= this.radius;
  }
  /**
   * Clamps a point within the sphere. If the point is outside the sphere, it
   * will clamp it to the closest point on the edge of the sphere. Points
   * already inside the sphere will not be affected.
   *
   * @param {Vector3} point - The plane to clamp.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The clamped point.
   */
  clampPoint(e, t) {
    const i = this.center.distanceToSquared(e);
    return t.copy(e), i > this.radius * this.radius && (t.sub(this.center).normalize(), t.multiplyScalar(this.radius).add(this.center)), t;
  }
  /**
   * Returns a bounding box that encloses this sphere.
   *
   * @param {Box3} target - The target box that is used to store the method's result.
   * @return {Box3} The bounding box that encloses this sphere.
   */
  getBoundingBox(e) {
    return this.isEmpty() ? (e.makeEmpty(), e) : (e.set(this.center, this.center), e.expandByScalar(this.radius), e);
  }
  /**
   * Transforms this sphere with the given 4x4 transformation matrix.
   *
   * @param {Matrix4} matrix - The transformation matrix.
   * @return {Sphere} A reference to this sphere.
   */
  applyMatrix4(e) {
    return this.center.applyMatrix4(e), this.radius = this.radius * e.getMaxScaleOnAxis(), this;
  }
  /**
   * Translates the sphere's center by the given offset.
   *
   * @param {Vector3} offset - The offset.
   * @return {Sphere} A reference to this sphere.
   */
  translate(e) {
    return this.center.add(e), this;
  }
  /**
   * Expands the boundaries of this sphere to include the given point.
   *
   * @param {Vector3} point - The point to include.
   * @return {Sphere} A reference to this sphere.
   */
  expandByPoint(e) {
    if (this.isEmpty())
      return this.center.copy(e), this.radius = 0, this;
    Zs.subVectors(e, this.center);
    const t = Zs.lengthSq();
    if (t > this.radius * this.radius) {
      const i = Math.sqrt(t), s = (i - this.radius) * 0.5;
      this.center.addScaledVector(Zs, s / i), this.radius += s;
    }
    return this;
  }
  /**
   * Expands this sphere to enclose both the original sphere and the given sphere.
   *
   * @param {Sphere} sphere - The sphere to include.
   * @return {Sphere} A reference to this sphere.
   */
  union(e) {
    return e.isEmpty() ? this : this.isEmpty() ? (this.copy(e), this) : (this.center.equals(e.center) === !0 ? this.radius = Math.max(this.radius, e.radius) : (Aa.subVectors(e.center, this.center).setLength(e.radius), this.expandByPoint(Zs.copy(e.center).add(Aa)), this.expandByPoint(Zs.copy(e.center).sub(Aa))), this);
  }
  /**
   * Returns `true` if this sphere is equal with the given one.
   *
   * @param {Sphere} sphere - The sphere to test for equality.
   * @return {boolean} Whether this bounding sphere is equal with the given one.
   */
  equals(e) {
    return e.center.equals(this.center) && e.radius === this.radius;
  }
  /**
   * Returns a new sphere with copied values from this instance.
   *
   * @return {Sphere} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Returns a serialized structure of the bounding sphere.
   *
   * @return {Object} Serialized structure with fields representing the object state.
   */
  toJSON() {
    return {
      radius: this.radius,
      center: this.center.toArray()
    };
  }
  /**
   * Returns a serialized structure of the bounding sphere.
   *
   * @param {Object} json - The serialized json to set the sphere from.
   * @return {Box3} A reference to this bounding sphere.
   */
  fromJSON(e) {
    return this.radius = e.radius, this.center.fromArray(e.center), this;
  }
}
const Wn = /* @__PURE__ */ new N(), wa = /* @__PURE__ */ new N(), Xr = /* @__PURE__ */ new N(), ci = /* @__PURE__ */ new N(), Ra = /* @__PURE__ */ new N(), Yr = /* @__PURE__ */ new N(), Ca = /* @__PURE__ */ new N();
class na {
  /**
   * Constructs a new ray.
   *
   * @param {Vector3} [origin=(0,0,0)] - The origin of the ray.
   * @param {Vector3} [direction=(0,0,-1)] - The (normalized) direction of the ray.
   */
  constructor(e = new N(), t = new N(0, 0, -1)) {
    this.origin = e, this.direction = t;
  }
  /**
   * Sets the ray's components by copying the given values.
   *
   * @param {Vector3} origin - The origin.
   * @param {Vector3} direction - The direction.
   * @return {Ray} A reference to this ray.
   */
  set(e, t) {
    return this.origin.copy(e), this.direction.copy(t), this;
  }
  /**
   * Copies the values of the given ray to this instance.
   *
   * @param {Ray} ray - The ray to copy.
   * @return {Ray} A reference to this ray.
   */
  copy(e) {
    return this.origin.copy(e.origin), this.direction.copy(e.direction), this;
  }
  /**
   * Returns a vector that is located at a given distance along this ray.
   *
   * @param {number} t - The distance along the ray to retrieve a position for.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} A position on the ray.
   */
  at(e, t) {
    return t.copy(this.origin).addScaledVector(this.direction, e);
  }
  /**
   * Adjusts the direction of the ray to point at the given vector in world space.
   *
   * @param {Vector3} v - The target position.
   * @return {Ray} A reference to this ray.
   */
  lookAt(e) {
    return this.direction.copy(e).sub(this.origin).normalize(), this;
  }
  /**
   * Shift the origin of this ray along its direction by the given distance.
   *
   * @param {number} t - The distance along the ray to interpolate.
   * @return {Ray} A reference to this ray.
   */
  recast(e) {
    return this.origin.copy(this.at(e, Wn)), this;
  }
  /**
   * Returns the point along this ray that is closest to the given point.
   *
   * @param {Vector3} point - A point in 3D space to get the closet location on the ray for.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The closest point on this ray.
   */
  closestPointToPoint(e, t) {
    t.subVectors(e, this.origin);
    const i = t.dot(this.direction);
    return i < 0 ? t.copy(this.origin) : t.copy(this.origin).addScaledVector(this.direction, i);
  }
  /**
   * Returns the distance of the closest approach between this ray and the given point.
   *
   * @param {Vector3} point - A point in 3D space to compute the distance to.
   * @return {number} The distance.
   */
  distanceToPoint(e) {
    return Math.sqrt(this.distanceSqToPoint(e));
  }
  /**
   * Returns the squared distance of the closest approach between this ray and the given point.
   *
   * @param {Vector3} point - A point in 3D space to compute the distance to.
   * @return {number} The squared distance.
   */
  distanceSqToPoint(e) {
    const t = Wn.subVectors(e, this.origin).dot(this.direction);
    return t < 0 ? this.origin.distanceToSquared(e) : (Wn.copy(this.origin).addScaledVector(this.direction, t), Wn.distanceToSquared(e));
  }
  /**
   * Returns the squared distance between this ray and the given line segment.
   *
   * @param {Vector3} v0 - The start point of the line segment.
   * @param {Vector3} v1 - The end point of the line segment.
   * @param {Vector3} [optionalPointOnRay] - When provided, it receives the point on this ray that is closest to the segment.
   * @param {Vector3} [optionalPointOnSegment] - When provided, it receives the point on the line segment that is closest to this ray.
   * @return {number} The squared distance.
   */
  distanceSqToSegment(e, t, i, s) {
    wa.copy(e).add(t).multiplyScalar(0.5), Xr.copy(t).sub(e).normalize(), ci.copy(this.origin).sub(wa);
    const r = e.distanceTo(t) * 0.5, o = -this.direction.dot(Xr), a = ci.dot(this.direction), l = -ci.dot(Xr), c = ci.lengthSq(), u = Math.abs(1 - o * o);
    let h, f, p, v;
    if (u > 0)
      if (h = o * l - a, f = o * a - l, v = r * u, h >= 0)
        if (f >= -v)
          if (f <= v) {
            const x = 1 / u;
            h *= x, f *= x, p = h * (h + o * f + 2 * a) + f * (o * h + f + 2 * l) + c;
          } else
            f = r, h = Math.max(0, -(o * f + a)), p = -h * h + f * (f + 2 * l) + c;
        else
          f = -r, h = Math.max(0, -(o * f + a)), p = -h * h + f * (f + 2 * l) + c;
      else
        f <= -v ? (h = Math.max(0, -(-o * r + a)), f = h > 0 ? -r : Math.min(Math.max(-r, -l), r), p = -h * h + f * (f + 2 * l) + c) : f <= v ? (h = 0, f = Math.min(Math.max(-r, -l), r), p = f * (f + 2 * l) + c) : (h = Math.max(0, -(o * r + a)), f = h > 0 ? r : Math.min(Math.max(-r, -l), r), p = -h * h + f * (f + 2 * l) + c);
    else
      f = o > 0 ? -r : r, h = Math.max(0, -(o * f + a)), p = -h * h + f * (f + 2 * l) + c;
    return i && i.copy(this.origin).addScaledVector(this.direction, h), s && s.copy(wa).addScaledVector(Xr, f), p;
  }
  /**
   * Intersects this ray with the given sphere, returning the intersection
   * point or `null` if there is no intersection.
   *
   * @param {Sphere} sphere - The sphere to intersect.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectSphere(e, t) {
    Wn.subVectors(e.center, this.origin);
    const i = Wn.dot(this.direction), s = Wn.dot(Wn) - i * i, r = e.radius * e.radius;
    if (s > r) return null;
    const o = Math.sqrt(r - s), a = i - o, l = i + o;
    return l < 0 ? null : a < 0 ? this.at(l, t) : this.at(a, t);
  }
  /**
   * Returns `true` if this ray intersects with the given sphere.
   *
   * @param {Sphere} sphere - The sphere to intersect.
   * @return {boolean} Whether this ray intersects with the given sphere or not.
   */
  intersectsSphere(e) {
    return e.radius < 0 ? !1 : this.distanceSqToPoint(e.center) <= e.radius * e.radius;
  }
  /**
   * Computes the distance from the ray's origin to the given plane. Returns `null` if the ray
   * does not intersect with the plane.
   *
   * @param {Plane} plane - The plane to compute the distance to.
   * @return {?number} Whether this ray intersects with the given sphere or not.
   */
  distanceToPlane(e) {
    const t = e.normal.dot(this.direction);
    if (t === 0)
      return e.distanceToPoint(this.origin) === 0 ? 0 : null;
    const i = -(this.origin.dot(e.normal) + e.constant) / t;
    return i >= 0 ? i : null;
  }
  /**
   * Intersects this ray with the given plane, returning the intersection
   * point or `null` if there is no intersection.
   *
   * @param {Plane} plane - The plane to intersect.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectPlane(e, t) {
    const i = this.distanceToPlane(e);
    return i === null ? null : this.at(i, t);
  }
  /**
   * Returns `true` if this ray intersects with the given plane.
   *
   * @param {Plane} plane - The plane to intersect.
   * @return {boolean} Whether this ray intersects with the given plane or not.
   */
  intersectsPlane(e) {
    const t = e.distanceToPoint(this.origin);
    return t === 0 || e.normal.dot(this.direction) * t < 0;
  }
  /**
   * Intersects this ray with the given bounding box, returning the intersection
   * point or `null` if there is no intersection.
   *
   * @param {Box3} box - The box to intersect.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectBox(e, t) {
    let i, s, r, o, a, l;
    const c = 1 / this.direction.x, u = 1 / this.direction.y, h = 1 / this.direction.z, f = this.origin;
    return c >= 0 ? (i = (e.min.x - f.x) * c, s = (e.max.x - f.x) * c) : (i = (e.max.x - f.x) * c, s = (e.min.x - f.x) * c), u >= 0 ? (r = (e.min.y - f.y) * u, o = (e.max.y - f.y) * u) : (r = (e.max.y - f.y) * u, o = (e.min.y - f.y) * u), i > o || r > s || ((r > i || isNaN(i)) && (i = r), (o < s || isNaN(s)) && (s = o), h >= 0 ? (a = (e.min.z - f.z) * h, l = (e.max.z - f.z) * h) : (a = (e.max.z - f.z) * h, l = (e.min.z - f.z) * h), i > l || a > s) || ((a > i || i !== i) && (i = a), (l < s || s !== s) && (s = l), s < 0) ? null : this.at(i >= 0 ? i : s, t);
  }
  /**
   * Returns `true` if this ray intersects with the given box.
   *
   * @param {Box3} box - The box to intersect.
   * @return {boolean} Whether this ray intersects with the given box or not.
   */
  intersectsBox(e) {
    return this.intersectBox(e, Wn) !== null;
  }
  /**
   * Intersects this ray with the given triangle, returning the intersection
   * point or `null` if there is no intersection.
   *
   * @param {Vector3} a - The first vertex of the triangle.
   * @param {Vector3} b - The second vertex of the triangle.
   * @param {Vector3} c - The third vertex of the triangle.
   * @param {boolean} backfaceCulling - Whether to use backface culling or not.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectTriangle(e, t, i, s, r) {
    Ra.subVectors(t, e), Yr.subVectors(i, e), Ca.crossVectors(Ra, Yr);
    let o = this.direction.dot(Ca), a;
    if (o > 0) {
      if (s) return null;
      a = 1;
    } else if (o < 0)
      a = -1, o = -o;
    else
      return null;
    ci.subVectors(this.origin, e);
    const l = a * this.direction.dot(Yr.crossVectors(ci, Yr));
    if (l < 0)
      return null;
    const c = a * this.direction.dot(Ra.cross(ci));
    if (c < 0 || l + c > o)
      return null;
    const u = -a * ci.dot(Ca);
    return u < 0 ? null : this.at(u / o, r);
  }
  /**
   * Transforms this ray with the given 4x4 transformation matrix.
   *
   * @param {Matrix4} matrix4 - The transformation matrix.
   * @return {Ray} A reference to this ray.
   */
  applyMatrix4(e) {
    return this.origin.applyMatrix4(e), this.direction.transformDirection(e), this;
  }
  /**
   * Returns `true` if this ray is equal with the given one.
   *
   * @param {Ray} ray - The ray to test for equality.
   * @return {boolean} Whether this ray is equal with the given one.
   */
  equals(e) {
    return e.origin.equals(this.origin) && e.direction.equals(this.direction);
  }
  /**
   * Returns a new ray with copied values from this instance.
   *
   * @return {Ray} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
}
class pt {
  /**
   * Constructs a new 4x4 matrix. The arguments are supposed to be
   * in row-major order. If no arguments are provided, the constructor
   * initializes the matrix as an identity matrix.
   *
   * @param {number} [n11] - 1-1 matrix element.
   * @param {number} [n12] - 1-2 matrix element.
   * @param {number} [n13] - 1-3 matrix element.
   * @param {number} [n14] - 1-4 matrix element.
   * @param {number} [n21] - 2-1 matrix element.
   * @param {number} [n22] - 2-2 matrix element.
   * @param {number} [n23] - 2-3 matrix element.
   * @param {number} [n24] - 2-4 matrix element.
   * @param {number} [n31] - 3-1 matrix element.
   * @param {number} [n32] - 3-2 matrix element.
   * @param {number} [n33] - 3-3 matrix element.
   * @param {number} [n34] - 3-4 matrix element.
   * @param {number} [n41] - 4-1 matrix element.
   * @param {number} [n42] - 4-2 matrix element.
   * @param {number} [n43] - 4-3 matrix element.
   * @param {number} [n44] - 4-4 matrix element.
   */
  constructor(e, t, i, s, r, o, a, l, c, u, h, f, p, v, x, m) {
    pt.prototype.isMatrix4 = !0, this.elements = [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ], e !== void 0 && this.set(e, t, i, s, r, o, a, l, c, u, h, f, p, v, x, m);
  }
  /**
   * Sets the elements of the matrix.The arguments are supposed to be
   * in row-major order.
   *
   * @param {number} [n11] - 1-1 matrix element.
   * @param {number} [n12] - 1-2 matrix element.
   * @param {number} [n13] - 1-3 matrix element.
   * @param {number} [n14] - 1-4 matrix element.
   * @param {number} [n21] - 2-1 matrix element.
   * @param {number} [n22] - 2-2 matrix element.
   * @param {number} [n23] - 2-3 matrix element.
   * @param {number} [n24] - 2-4 matrix element.
   * @param {number} [n31] - 3-1 matrix element.
   * @param {number} [n32] - 3-2 matrix element.
   * @param {number} [n33] - 3-3 matrix element.
   * @param {number} [n34] - 3-4 matrix element.
   * @param {number} [n41] - 4-1 matrix element.
   * @param {number} [n42] - 4-2 matrix element.
   * @param {number} [n43] - 4-3 matrix element.
   * @param {number} [n44] - 4-4 matrix element.
   * @return {Matrix4} A reference to this matrix.
   */
  set(e, t, i, s, r, o, a, l, c, u, h, f, p, v, x, m) {
    const d = this.elements;
    return d[0] = e, d[4] = t, d[8] = i, d[12] = s, d[1] = r, d[5] = o, d[9] = a, d[13] = l, d[2] = c, d[6] = u, d[10] = h, d[14] = f, d[3] = p, d[7] = v, d[11] = x, d[15] = m, this;
  }
  /**
   * Sets this matrix to the 4x4 identity matrix.
   *
   * @return {Matrix4} A reference to this matrix.
   */
  identity() {
    return this.set(
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Returns a matrix with copied values from this instance.
   *
   * @return {Matrix4} A clone of this instance.
   */
  clone() {
    return new pt().fromArray(this.elements);
  }
  /**
   * Copies the values of the given matrix to this instance.
   *
   * @param {Matrix4} m - The matrix to copy.
   * @return {Matrix4} A reference to this matrix.
   */
  copy(e) {
    const t = this.elements, i = e.elements;
    return t[0] = i[0], t[1] = i[1], t[2] = i[2], t[3] = i[3], t[4] = i[4], t[5] = i[5], t[6] = i[6], t[7] = i[7], t[8] = i[8], t[9] = i[9], t[10] = i[10], t[11] = i[11], t[12] = i[12], t[13] = i[13], t[14] = i[14], t[15] = i[15], this;
  }
  /**
   * Copies the translation component of the given matrix
   * into this matrix's translation component.
   *
   * @param {Matrix4} m - The matrix to copy the translation component.
   * @return {Matrix4} A reference to this matrix.
   */
  copyPosition(e) {
    const t = this.elements, i = e.elements;
    return t[12] = i[12], t[13] = i[13], t[14] = i[14], this;
  }
  /**
   * Set the upper 3x3 elements of this matrix to the values of given 3x3 matrix.
   *
   * @param {Matrix3} m - The 3x3 matrix.
   * @return {Matrix4} A reference to this matrix.
   */
  setFromMatrix3(e) {
    const t = e.elements;
    return this.set(
      t[0],
      t[3],
      t[6],
      0,
      t[1],
      t[4],
      t[7],
      0,
      t[2],
      t[5],
      t[8],
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Extracts the basis of this matrix into the three axis vectors provided.
   *
   * @param {Vector3} xAxis - The basis's x axis.
   * @param {Vector3} yAxis - The basis's y axis.
   * @param {Vector3} zAxis - The basis's z axis.
   * @return {Matrix4} A reference to this matrix.
   */
  extractBasis(e, t, i) {
    return e.setFromMatrixColumn(this, 0), t.setFromMatrixColumn(this, 1), i.setFromMatrixColumn(this, 2), this;
  }
  /**
   * Sets the given basis vectors to this matrix.
   *
   * @param {Vector3} xAxis - The basis's x axis.
   * @param {Vector3} yAxis - The basis's y axis.
   * @param {Vector3} zAxis - The basis's z axis.
   * @return {Matrix4} A reference to this matrix.
   */
  makeBasis(e, t, i) {
    return this.set(
      e.x,
      t.x,
      i.x,
      0,
      e.y,
      t.y,
      i.y,
      0,
      e.z,
      t.z,
      i.z,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Extracts the rotation component of the given matrix
   * into this matrix's rotation component.
   *
   * Note: This method does not support reflection matrices.
   *
   * @param {Matrix4} m - The matrix.
   * @return {Matrix4} A reference to this matrix.
   */
  extractRotation(e) {
    const t = this.elements, i = e.elements, s = 1 / as.setFromMatrixColumn(e, 0).length(), r = 1 / as.setFromMatrixColumn(e, 1).length(), o = 1 / as.setFromMatrixColumn(e, 2).length();
    return t[0] = i[0] * s, t[1] = i[1] * s, t[2] = i[2] * s, t[3] = 0, t[4] = i[4] * r, t[5] = i[5] * r, t[6] = i[6] * r, t[7] = 0, t[8] = i[8] * o, t[9] = i[9] * o, t[10] = i[10] * o, t[11] = 0, t[12] = 0, t[13] = 0, t[14] = 0, t[15] = 1, this;
  }
  /**
   * Sets the rotation component (the upper left 3x3 matrix) of this matrix to
   * the rotation specified by the given Euler angles. The rest of
   * the matrix is set to the identity. Depending on the {@link Euler#order},
   * there are six possible outcomes. See [this page]{@link https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix}
   * for a complete list.
   *
   * @param {Euler} euler - The Euler angles.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationFromEuler(e) {
    const t = this.elements, i = e.x, s = e.y, r = e.z, o = Math.cos(i), a = Math.sin(i), l = Math.cos(s), c = Math.sin(s), u = Math.cos(r), h = Math.sin(r);
    if (e.order === "XYZ") {
      const f = o * u, p = o * h, v = a * u, x = a * h;
      t[0] = l * u, t[4] = -l * h, t[8] = c, t[1] = p + v * c, t[5] = f - x * c, t[9] = -a * l, t[2] = x - f * c, t[6] = v + p * c, t[10] = o * l;
    } else if (e.order === "YXZ") {
      const f = l * u, p = l * h, v = c * u, x = c * h;
      t[0] = f + x * a, t[4] = v * a - p, t[8] = o * c, t[1] = o * h, t[5] = o * u, t[9] = -a, t[2] = p * a - v, t[6] = x + f * a, t[10] = o * l;
    } else if (e.order === "ZXY") {
      const f = l * u, p = l * h, v = c * u, x = c * h;
      t[0] = f - x * a, t[4] = -o * h, t[8] = v + p * a, t[1] = p + v * a, t[5] = o * u, t[9] = x - f * a, t[2] = -o * c, t[6] = a, t[10] = o * l;
    } else if (e.order === "ZYX") {
      const f = o * u, p = o * h, v = a * u, x = a * h;
      t[0] = l * u, t[4] = v * c - p, t[8] = f * c + x, t[1] = l * h, t[5] = x * c + f, t[9] = p * c - v, t[2] = -c, t[6] = a * l, t[10] = o * l;
    } else if (e.order === "YZX") {
      const f = o * l, p = o * c, v = a * l, x = a * c;
      t[0] = l * u, t[4] = x - f * h, t[8] = v * h + p, t[1] = h, t[5] = o * u, t[9] = -a * u, t[2] = -c * u, t[6] = p * h + v, t[10] = f - x * h;
    } else if (e.order === "XZY") {
      const f = o * l, p = o * c, v = a * l, x = a * c;
      t[0] = l * u, t[4] = -h, t[8] = c * u, t[1] = f * h + x, t[5] = o * u, t[9] = p * h - v, t[2] = v * h - p, t[6] = a * u, t[10] = x * h + f;
    }
    return t[3] = 0, t[7] = 0, t[11] = 0, t[12] = 0, t[13] = 0, t[14] = 0, t[15] = 1, this;
  }
  /**
   * Sets the rotation component of this matrix to the rotation specified by
   * the given Quaternion as outlined [here]{@link https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion}
   * The rest of the matrix is set to the identity.
   *
   * @param {Quaternion} q - The Quaternion.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationFromQuaternion(e) {
    return this.compose(Bg, e, zg);
  }
  /**
   * Sets the rotation component of the transformation matrix, looking from `eye` towards
   * `target`, and oriented by the up-direction.
   *
   * @param {Vector3} eye - The eye vector.
   * @param {Vector3} target - The target vector.
   * @param {Vector3} up - The up vector.
   * @return {Matrix4} A reference to this matrix.
   */
  lookAt(e, t, i) {
    const s = this.elements;
    return Qt.subVectors(e, t), Qt.lengthSq() === 0 && (Qt.z = 1), Qt.normalize(), ui.crossVectors(i, Qt), ui.lengthSq() === 0 && (Math.abs(i.z) === 1 ? Qt.x += 1e-4 : Qt.z += 1e-4, Qt.normalize(), ui.crossVectors(i, Qt)), ui.normalize(), qr.crossVectors(Qt, ui), s[0] = ui.x, s[4] = qr.x, s[8] = Qt.x, s[1] = ui.y, s[5] = qr.y, s[9] = Qt.y, s[2] = ui.z, s[6] = qr.z, s[10] = Qt.z, this;
  }
  /**
   * Post-multiplies this matrix by the given 4x4 matrix.
   *
   * @param {Matrix4} m - The matrix to multiply with.
   * @return {Matrix4} A reference to this matrix.
   */
  multiply(e) {
    return this.multiplyMatrices(this, e);
  }
  /**
   * Pre-multiplies this matrix by the given 4x4 matrix.
   *
   * @param {Matrix4} m - The matrix to multiply with.
   * @return {Matrix4} A reference to this matrix.
   */
  premultiply(e) {
    return this.multiplyMatrices(e, this);
  }
  /**
   * Multiples the given 4x4 matrices and stores the result
   * in this matrix.
   *
   * @param {Matrix4} a - The first matrix.
   * @param {Matrix4} b - The second matrix.
   * @return {Matrix4} A reference to this matrix.
   */
  multiplyMatrices(e, t) {
    const i = e.elements, s = t.elements, r = this.elements, o = i[0], a = i[4], l = i[8], c = i[12], u = i[1], h = i[5], f = i[9], p = i[13], v = i[2], x = i[6], m = i[10], d = i[14], b = i[3], A = i[7], M = i[11], R = i[15], w = s[0], D = s[4], U = s[8], y = s[12], S = s[1], P = s[5], L = s[9], V = s[13], Z = s[2], te = s[6], $ = s[10], ie = s[14], H = s[3], fe = s[7], xe = s[11], me = s[15];
    return r[0] = o * w + a * S + l * Z + c * H, r[4] = o * D + a * P + l * te + c * fe, r[8] = o * U + a * L + l * $ + c * xe, r[12] = o * y + a * V + l * ie + c * me, r[1] = u * w + h * S + f * Z + p * H, r[5] = u * D + h * P + f * te + p * fe, r[9] = u * U + h * L + f * $ + p * xe, r[13] = u * y + h * V + f * ie + p * me, r[2] = v * w + x * S + m * Z + d * H, r[6] = v * D + x * P + m * te + d * fe, r[10] = v * U + x * L + m * $ + d * xe, r[14] = v * y + x * V + m * ie + d * me, r[3] = b * w + A * S + M * Z + R * H, r[7] = b * D + A * P + M * te + R * fe, r[11] = b * U + A * L + M * $ + R * xe, r[15] = b * y + A * V + M * ie + R * me, this;
  }
  /**
   * Multiplies every component of the matrix by the given scalar.
   *
   * @param {number} s - The scalar.
   * @return {Matrix4} A reference to this matrix.
   */
  multiplyScalar(e) {
    const t = this.elements;
    return t[0] *= e, t[4] *= e, t[8] *= e, t[12] *= e, t[1] *= e, t[5] *= e, t[9] *= e, t[13] *= e, t[2] *= e, t[6] *= e, t[10] *= e, t[14] *= e, t[3] *= e, t[7] *= e, t[11] *= e, t[15] *= e, this;
  }
  /**
   * Computes and returns the determinant of this matrix.
   *
   * Based on the method outlined [here]{@link http://www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.html}.
   *
   * @return {number} The determinant.
   */
  determinant() {
    const e = this.elements, t = e[0], i = e[4], s = e[8], r = e[12], o = e[1], a = e[5], l = e[9], c = e[13], u = e[2], h = e[6], f = e[10], p = e[14], v = e[3], x = e[7], m = e[11], d = e[15];
    return v * (+r * l * h - s * c * h - r * a * f + i * c * f + s * a * p - i * l * p) + x * (+t * l * p - t * c * f + r * o * f - s * o * p + s * c * u - r * l * u) + m * (+t * c * h - t * a * p - r * o * h + i * o * p + r * a * u - i * c * u) + d * (-s * a * u - t * l * h + t * a * f + s * o * h - i * o * f + i * l * u);
  }
  /**
   * Transposes this matrix in place.
   *
   * @return {Matrix4} A reference to this matrix.
   */
  transpose() {
    const e = this.elements;
    let t;
    return t = e[1], e[1] = e[4], e[4] = t, t = e[2], e[2] = e[8], e[8] = t, t = e[6], e[6] = e[9], e[9] = t, t = e[3], e[3] = e[12], e[12] = t, t = e[7], e[7] = e[13], e[13] = t, t = e[11], e[11] = e[14], e[14] = t, this;
  }
  /**
   * Sets the position component for this matrix from the given vector,
   * without affecting the rest of the matrix.
   *
   * @param {number|Vector3} x - The x component of the vector or alternatively the vector object.
   * @param {number} y - The y component of the vector.
   * @param {number} z - The z component of the vector.
   * @return {Matrix4} A reference to this matrix.
   */
  setPosition(e, t, i) {
    const s = this.elements;
    return e.isVector3 ? (s[12] = e.x, s[13] = e.y, s[14] = e.z) : (s[12] = e, s[13] = t, s[14] = i), this;
  }
  /**
   * Inverts this matrix, using the [analytic method]{@link https://en.wikipedia.org/wiki/Invertible_matrix#Analytic_solution}.
   * You can not invert with a determinant of zero. If you attempt this, the method produces
   * a zero matrix instead.
   *
   * @return {Matrix4} A reference to this matrix.
   */
  invert() {
    const e = this.elements, t = e[0], i = e[1], s = e[2], r = e[3], o = e[4], a = e[5], l = e[6], c = e[7], u = e[8], h = e[9], f = e[10], p = e[11], v = e[12], x = e[13], m = e[14], d = e[15], b = h * m * c - x * f * c + x * l * p - a * m * p - h * l * d + a * f * d, A = v * f * c - u * m * c - v * l * p + o * m * p + u * l * d - o * f * d, M = u * x * c - v * h * c + v * a * p - o * x * p - u * a * d + o * h * d, R = v * h * l - u * x * l - v * a * f + o * x * f + u * a * m - o * h * m, w = t * b + i * A + s * M + r * R;
    if (w === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const D = 1 / w;
    return e[0] = b * D, e[1] = (x * f * r - h * m * r - x * s * p + i * m * p + h * s * d - i * f * d) * D, e[2] = (a * m * r - x * l * r + x * s * c - i * m * c - a * s * d + i * l * d) * D, e[3] = (h * l * r - a * f * r - h * s * c + i * f * c + a * s * p - i * l * p) * D, e[4] = A * D, e[5] = (u * m * r - v * f * r + v * s * p - t * m * p - u * s * d + t * f * d) * D, e[6] = (v * l * r - o * m * r - v * s * c + t * m * c + o * s * d - t * l * d) * D, e[7] = (o * f * r - u * l * r + u * s * c - t * f * c - o * s * p + t * l * p) * D, e[8] = M * D, e[9] = (v * h * r - u * x * r - v * i * p + t * x * p + u * i * d - t * h * d) * D, e[10] = (o * x * r - v * a * r + v * i * c - t * x * c - o * i * d + t * a * d) * D, e[11] = (u * a * r - o * h * r - u * i * c + t * h * c + o * i * p - t * a * p) * D, e[12] = R * D, e[13] = (u * x * s - v * h * s + v * i * f - t * x * f - u * i * m + t * h * m) * D, e[14] = (v * a * s - o * x * s - v * i * l + t * x * l + o * i * m - t * a * m) * D, e[15] = (o * h * s - u * a * s + u * i * l - t * h * l - o * i * f + t * a * f) * D, this;
  }
  /**
   * Multiplies the columns of this matrix by the given vector.
   *
   * @param {Vector3} v - The scale vector.
   * @return {Matrix4} A reference to this matrix.
   */
  scale(e) {
    const t = this.elements, i = e.x, s = e.y, r = e.z;
    return t[0] *= i, t[4] *= s, t[8] *= r, t[1] *= i, t[5] *= s, t[9] *= r, t[2] *= i, t[6] *= s, t[10] *= r, t[3] *= i, t[7] *= s, t[11] *= r, this;
  }
  /**
   * Gets the maximum scale value of the three axes.
   *
   * @return {number} The maximum scale.
   */
  getMaxScaleOnAxis() {
    const e = this.elements, t = e[0] * e[0] + e[1] * e[1] + e[2] * e[2], i = e[4] * e[4] + e[5] * e[5] + e[6] * e[6], s = e[8] * e[8] + e[9] * e[9] + e[10] * e[10];
    return Math.sqrt(Math.max(t, i, s));
  }
  /**
   * Sets this matrix as a translation transform from the given vector.
   *
   * @param {number|Vector3} x - The amount to translate in the X axis or alternatively a translation vector.
   * @param {number} y - The amount to translate in the Y axis.
   * @param {number} z - The amount to translate in the z axis.
   * @return {Matrix4} A reference to this matrix.
   */
  makeTranslation(e, t, i) {
    return e.isVector3 ? this.set(
      1,
      0,
      0,
      e.x,
      0,
      1,
      0,
      e.y,
      0,
      0,
      1,
      e.z,
      0,
      0,
      0,
      1
    ) : this.set(
      1,
      0,
      0,
      e,
      0,
      1,
      0,
      t,
      0,
      0,
      1,
      i,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a rotational transformation around the X axis by
   * the given angle.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationX(e) {
    const t = Math.cos(e), i = Math.sin(e);
    return this.set(
      1,
      0,
      0,
      0,
      0,
      t,
      -i,
      0,
      0,
      i,
      t,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a rotational transformation around the Y axis by
   * the given angle.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationY(e) {
    const t = Math.cos(e), i = Math.sin(e);
    return this.set(
      t,
      0,
      i,
      0,
      0,
      1,
      0,
      0,
      -i,
      0,
      t,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a rotational transformation around the Z axis by
   * the given angle.
   *
   * @param {number} theta - The rotation in radians.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationZ(e) {
    const t = Math.cos(e), i = Math.sin(e);
    return this.set(
      t,
      -i,
      0,
      0,
      i,
      t,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a rotational transformation around the given axis by
   * the given angle.
   *
   * This is a somewhat controversial but mathematically sound alternative to
   * rotating via Quaternions. See the discussion [here]{@link https://www.gamedev.net/articles/programming/math-and-physics/do-we-really-need-quaternions-r1199}.
   *
   * @param {Vector3} axis - The normalized rotation axis.
   * @param {number} angle - The rotation in radians.
   * @return {Matrix4} A reference to this matrix.
   */
  makeRotationAxis(e, t) {
    const i = Math.cos(t), s = Math.sin(t), r = 1 - i, o = e.x, a = e.y, l = e.z, c = r * o, u = r * a;
    return this.set(
      c * o + i,
      c * a - s * l,
      c * l + s * a,
      0,
      c * a + s * l,
      u * a + i,
      u * l - s * o,
      0,
      c * l - s * a,
      u * l + s * o,
      r * l * l + i,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a scale transformation.
   *
   * @param {number} x - The amount to scale in the X axis.
   * @param {number} y - The amount to scale in the Y axis.
   * @param {number} z - The amount to scale in the Z axis.
   * @return {Matrix4} A reference to this matrix.
   */
  makeScale(e, t, i) {
    return this.set(
      e,
      0,
      0,
      0,
      0,
      t,
      0,
      0,
      0,
      0,
      i,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix as a shear transformation.
   *
   * @param {number} xy - The amount to shear X by Y.
   * @param {number} xz - The amount to shear X by Z.
   * @param {number} yx - The amount to shear Y by X.
   * @param {number} yz - The amount to shear Y by Z.
   * @param {number} zx - The amount to shear Z by X.
   * @param {number} zy - The amount to shear Z by Y.
   * @return {Matrix4} A reference to this matrix.
   */
  makeShear(e, t, i, s, r, o) {
    return this.set(
      1,
      i,
      r,
      0,
      e,
      1,
      o,
      0,
      t,
      s,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  /**
   * Sets this matrix to the transformation composed of the given position,
   * rotation (Quaternion) and scale.
   *
   * @param {Vector3} position - The position vector.
   * @param {Quaternion} quaternion - The rotation as a Quaternion.
   * @param {Vector3} scale - The scale vector.
   * @return {Matrix4} A reference to this matrix.
   */
  compose(e, t, i) {
    const s = this.elements, r = t._x, o = t._y, a = t._z, l = t._w, c = r + r, u = o + o, h = a + a, f = r * c, p = r * u, v = r * h, x = o * u, m = o * h, d = a * h, b = l * c, A = l * u, M = l * h, R = i.x, w = i.y, D = i.z;
    return s[0] = (1 - (x + d)) * R, s[1] = (p + M) * R, s[2] = (v - A) * R, s[3] = 0, s[4] = (p - M) * w, s[5] = (1 - (f + d)) * w, s[6] = (m + b) * w, s[7] = 0, s[8] = (v + A) * D, s[9] = (m - b) * D, s[10] = (1 - (f + x)) * D, s[11] = 0, s[12] = e.x, s[13] = e.y, s[14] = e.z, s[15] = 1, this;
  }
  /**
   * Decomposes this matrix into its position, rotation and scale components
   * and provides the result in the given objects.
   *
   * Note: Not all matrices are decomposable in this way. For example, if an
   * object has a non-uniformly scaled parent, then the object's world matrix
   * may not be decomposable, and this method may not be appropriate.
   *
   * @param {Vector3} position - The position vector.
   * @param {Quaternion} quaternion - The rotation as a Quaternion.
   * @param {Vector3} scale - The scale vector.
   * @return {Matrix4} A reference to this matrix.
   */
  decompose(e, t, i) {
    const s = this.elements;
    let r = as.set(s[0], s[1], s[2]).length();
    const o = as.set(s[4], s[5], s[6]).length(), a = as.set(s[8], s[9], s[10]).length();
    this.determinant() < 0 && (r = -r), e.x = s[12], e.y = s[13], e.z = s[14], gn.copy(this);
    const c = 1 / r, u = 1 / o, h = 1 / a;
    return gn.elements[0] *= c, gn.elements[1] *= c, gn.elements[2] *= c, gn.elements[4] *= u, gn.elements[5] *= u, gn.elements[6] *= u, gn.elements[8] *= h, gn.elements[9] *= h, gn.elements[10] *= h, t.setFromRotationMatrix(gn), i.x = r, i.y = o, i.z = a, this;
  }
  /**
	 * Creates a perspective projection matrix. This is used internally by
	 * {@link PerspectiveCamera#updateProjectionMatrix}.

	 * @param {number} left - Left boundary of the viewing frustum at the near plane.
	 * @param {number} right - Right boundary of the viewing frustum at the near plane.
	 * @param {number} top - Top boundary of the viewing frustum at the near plane.
	 * @param {number} bottom - Bottom boundary of the viewing frustum at the near plane.
	 * @param {number} near - The distance from the camera to the near plane.
	 * @param {number} far - The distance from the camera to the far plane.
	 * @param {(WebGLCoordinateSystem|WebGPUCoordinateSystem)} [coordinateSystem=WebGLCoordinateSystem] - The coordinate system.
	 * @param {boolean} [reversedDepth=false] - Whether to use a reversed depth.
	 * @return {Matrix4} A reference to this matrix.
	 */
  makePerspective(e, t, i, s, r, o, a = Nn, l = !1) {
    const c = this.elements, u = 2 * r / (t - e), h = 2 * r / (i - s), f = (t + e) / (t - e), p = (i + s) / (i - s);
    let v, x;
    if (l)
      v = r / (o - r), x = o * r / (o - r);
    else if (a === Nn)
      v = -(o + r) / (o - r), x = -2 * o * r / (o - r);
    else if (a === zo)
      v = -o / (o - r), x = -o * r / (o - r);
    else
      throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: " + a);
    return c[0] = u, c[4] = 0, c[8] = f, c[12] = 0, c[1] = 0, c[5] = h, c[9] = p, c[13] = 0, c[2] = 0, c[6] = 0, c[10] = v, c[14] = x, c[3] = 0, c[7] = 0, c[11] = -1, c[15] = 0, this;
  }
  /**
	 * Creates a orthographic projection matrix. This is used internally by
	 * {@link OrthographicCamera#updateProjectionMatrix}.

	 * @param {number} left - Left boundary of the viewing frustum at the near plane.
	 * @param {number} right - Right boundary of the viewing frustum at the near plane.
	 * @param {number} top - Top boundary of the viewing frustum at the near plane.
	 * @param {number} bottom - Bottom boundary of the viewing frustum at the near plane.
	 * @param {number} near - The distance from the camera to the near plane.
	 * @param {number} far - The distance from the camera to the far plane.
	 * @param {(WebGLCoordinateSystem|WebGPUCoordinateSystem)} [coordinateSystem=WebGLCoordinateSystem] - The coordinate system.
	 * @param {boolean} [reversedDepth=false] - Whether to use a reversed depth.
	 * @return {Matrix4} A reference to this matrix.
	 */
  makeOrthographic(e, t, i, s, r, o, a = Nn, l = !1) {
    const c = this.elements, u = 2 / (t - e), h = 2 / (i - s), f = -(t + e) / (t - e), p = -(i + s) / (i - s);
    let v, x;
    if (l)
      v = 1 / (o - r), x = o / (o - r);
    else if (a === Nn)
      v = -2 / (o - r), x = -(o + r) / (o - r);
    else if (a === zo)
      v = -1 / (o - r), x = -r / (o - r);
    else
      throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: " + a);
    return c[0] = u, c[4] = 0, c[8] = 0, c[12] = f, c[1] = 0, c[5] = h, c[9] = 0, c[13] = p, c[2] = 0, c[6] = 0, c[10] = v, c[14] = x, c[3] = 0, c[7] = 0, c[11] = 0, c[15] = 1, this;
  }
  /**
   * Returns `true` if this matrix is equal with the given one.
   *
   * @param {Matrix4} matrix - The matrix to test for equality.
   * @return {boolean} Whether this matrix is equal with the given one.
   */
  equals(e) {
    const t = this.elements, i = e.elements;
    for (let s = 0; s < 16; s++)
      if (t[s] !== i[s]) return !1;
    return !0;
  }
  /**
   * Sets the elements of the matrix from the given array.
   *
   * @param {Array<number>} array - The matrix elements in column-major order.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Matrix4} A reference to this matrix.
   */
  fromArray(e, t = 0) {
    for (let i = 0; i < 16; i++)
      this.elements[i] = e[i + t];
    return this;
  }
  /**
   * Writes the elements of this matrix to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the matrix elements in column-major order.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The matrix elements in column-major order.
   */
  toArray(e = [], t = 0) {
    const i = this.elements;
    return e[t] = i[0], e[t + 1] = i[1], e[t + 2] = i[2], e[t + 3] = i[3], e[t + 4] = i[4], e[t + 5] = i[5], e[t + 6] = i[6], e[t + 7] = i[7], e[t + 8] = i[8], e[t + 9] = i[9], e[t + 10] = i[10], e[t + 11] = i[11], e[t + 12] = i[12], e[t + 13] = i[13], e[t + 14] = i[14], e[t + 15] = i[15], e;
  }
}
const as = /* @__PURE__ */ new N(), gn = /* @__PURE__ */ new pt(), Bg = /* @__PURE__ */ new N(0, 0, 0), zg = /* @__PURE__ */ new N(1, 1, 1), ui = /* @__PURE__ */ new N(), qr = /* @__PURE__ */ new N(), Qt = /* @__PURE__ */ new N(), Nu = /* @__PURE__ */ new pt(), Fu = /* @__PURE__ */ new Yi();
class zn {
  /**
   * Constructs a new euler instance.
   *
   * @param {number} [x=0] - The angle of the x axis in radians.
   * @param {number} [y=0] - The angle of the y axis in radians.
   * @param {number} [z=0] - The angle of the z axis in radians.
   * @param {string} [order=Euler.DEFAULT_ORDER] - A string representing the order that the rotations are applied.
   */
  constructor(e = 0, t = 0, i = 0, s = zn.DEFAULT_ORDER) {
    this.isEuler = !0, this._x = e, this._y = t, this._z = i, this._order = s;
  }
  /**
   * The angle of the x axis in radians.
   *
   * @type {number}
   * @default 0
   */
  get x() {
    return this._x;
  }
  set x(e) {
    this._x = e, this._onChangeCallback();
  }
  /**
   * The angle of the y axis in radians.
   *
   * @type {number}
   * @default 0
   */
  get y() {
    return this._y;
  }
  set y(e) {
    this._y = e, this._onChangeCallback();
  }
  /**
   * The angle of the z axis in radians.
   *
   * @type {number}
   * @default 0
   */
  get z() {
    return this._z;
  }
  set z(e) {
    this._z = e, this._onChangeCallback();
  }
  /**
   * A string representing the order that the rotations are applied.
   *
   * @type {string}
   * @default 'XYZ'
   */
  get order() {
    return this._order;
  }
  set order(e) {
    this._order = e, this._onChangeCallback();
  }
  /**
   * Sets the Euler components.
   *
   * @param {number} x - The angle of the x axis in radians.
   * @param {number} y - The angle of the y axis in radians.
   * @param {number} z - The angle of the z axis in radians.
   * @param {string} [order] - A string representing the order that the rotations are applied.
   * @return {Euler} A reference to this Euler instance.
   */
  set(e, t, i, s = this._order) {
    return this._x = e, this._y = t, this._z = i, this._order = s, this._onChangeCallback(), this;
  }
  /**
   * Returns a new Euler instance with copied values from this instance.
   *
   * @return {Euler} A clone of this instance.
   */
  clone() {
    return new this.constructor(this._x, this._y, this._z, this._order);
  }
  /**
   * Copies the values of the given Euler instance to this instance.
   *
   * @param {Euler} euler - The Euler instance to copy.
   * @return {Euler} A reference to this Euler instance.
   */
  copy(e) {
    return this._x = e._x, this._y = e._y, this._z = e._z, this._order = e._order, this._onChangeCallback(), this;
  }
  /**
   * Sets the angles of this Euler instance from a pure rotation matrix.
   *
   * @param {Matrix4} m - A 4x4 matrix of which the upper 3x3 of matrix is a pure rotation matrix (i.e. unscaled).
   * @param {string} [order] - A string representing the order that the rotations are applied.
   * @param {boolean} [update=true] - Whether the internal `onChange` callback should be executed or not.
   * @return {Euler} A reference to this Euler instance.
   */
  setFromRotationMatrix(e, t = this._order, i = !0) {
    const s = e.elements, r = s[0], o = s[4], a = s[8], l = s[1], c = s[5], u = s[9], h = s[2], f = s[6], p = s[10];
    switch (t) {
      case "XYZ":
        this._y = Math.asin(je(a, -1, 1)), Math.abs(a) < 0.9999999 ? (this._x = Math.atan2(-u, p), this._z = Math.atan2(-o, r)) : (this._x = Math.atan2(f, c), this._z = 0);
        break;
      case "YXZ":
        this._x = Math.asin(-je(u, -1, 1)), Math.abs(u) < 0.9999999 ? (this._y = Math.atan2(a, p), this._z = Math.atan2(l, c)) : (this._y = Math.atan2(-h, r), this._z = 0);
        break;
      case "ZXY":
        this._x = Math.asin(je(f, -1, 1)), Math.abs(f) < 0.9999999 ? (this._y = Math.atan2(-h, p), this._z = Math.atan2(-o, c)) : (this._y = 0, this._z = Math.atan2(l, r));
        break;
      case "ZYX":
        this._y = Math.asin(-je(h, -1, 1)), Math.abs(h) < 0.9999999 ? (this._x = Math.atan2(f, p), this._z = Math.atan2(l, r)) : (this._x = 0, this._z = Math.atan2(-o, c));
        break;
      case "YZX":
        this._z = Math.asin(je(l, -1, 1)), Math.abs(l) < 0.9999999 ? (this._x = Math.atan2(-u, c), this._y = Math.atan2(-h, r)) : (this._x = 0, this._y = Math.atan2(a, p));
        break;
      case "XZY":
        this._z = Math.asin(-je(o, -1, 1)), Math.abs(o) < 0.9999999 ? (this._x = Math.atan2(f, c), this._y = Math.atan2(a, r)) : (this._x = Math.atan2(-u, p), this._y = 0);
        break;
      default:
        console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: " + t);
    }
    return this._order = t, i === !0 && this._onChangeCallback(), this;
  }
  /**
   * Sets the angles of this Euler instance from a normalized quaternion.
   *
   * @param {Quaternion} q - A normalized Quaternion.
   * @param {string} [order] - A string representing the order that the rotations are applied.
   * @param {boolean} [update=true] - Whether the internal `onChange` callback should be executed or not.
   * @return {Euler} A reference to this Euler instance.
   */
  setFromQuaternion(e, t, i) {
    return Nu.makeRotationFromQuaternion(e), this.setFromRotationMatrix(Nu, t, i);
  }
  /**
   * Sets the angles of this Euler instance from the given vector.
   *
   * @param {Vector3} v - The vector.
   * @param {string} [order] - A string representing the order that the rotations are applied.
   * @return {Euler} A reference to this Euler instance.
   */
  setFromVector3(e, t = this._order) {
    return this.set(e.x, e.y, e.z, t);
  }
  /**
   * Resets the euler angle with a new order by creating a quaternion from this
   * euler angle and then setting this euler angle with the quaternion and the
   * new order.
   *
   * Warning: This discards revolution information.
   *
   * @param {string} [newOrder] - A string representing the new order that the rotations are applied.
   * @return {Euler} A reference to this Euler instance.
   */
  reorder(e) {
    return Fu.setFromEuler(this), this.setFromQuaternion(Fu, e);
  }
  /**
   * Returns `true` if this Euler instance is equal with the given one.
   *
   * @param {Euler} euler - The Euler instance to test for equality.
   * @return {boolean} Whether this Euler instance is equal with the given one.
   */
  equals(e) {
    return e._x === this._x && e._y === this._y && e._z === this._z && e._order === this._order;
  }
  /**
   * Sets this Euler instance's components to values from the given array. The first three
   * entries of the array are assign to the x,y and z components. An optional fourth entry
   * defines the Euler order.
   *
   * @param {Array<number,number,number,?string>} array - An array holding the Euler component values.
   * @return {Euler} A reference to this Euler instance.
   */
  fromArray(e) {
    return this._x = e[0], this._y = e[1], this._z = e[2], e[3] !== void 0 && (this._order = e[3]), this._onChangeCallback(), this;
  }
  /**
   * Writes the components of this Euler instance to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number,number,number,string>} [array=[]] - The target array holding the Euler components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number,number,number,string>} The Euler components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this._x, e[t + 1] = this._y, e[t + 2] = this._z, e[t + 3] = this._order, e;
  }
  _onChange(e) {
    return this._onChangeCallback = e, this;
  }
  _onChangeCallback() {
  }
  *[Symbol.iterator]() {
    yield this._x, yield this._y, yield this._z, yield this._order;
  }
}
zn.DEFAULT_ORDER = "XYZ";
class pd {
  /**
   * Constructs a new layers instance, with membership
   * initially set to layer `0`.
   */
  constructor() {
    this.mask = 1;
  }
  /**
   * Sets membership to the given layer, and remove membership all other layers.
   *
   * @param {number} layer - The layer to set.
   */
  set(e) {
    this.mask = (1 << e | 0) >>> 0;
  }
  /**
   * Adds membership of the given layer.
   *
   * @param {number} layer - The layer to enable.
   */
  enable(e) {
    this.mask |= 1 << e | 0;
  }
  /**
   * Adds membership to all layers.
   */
  enableAll() {
    this.mask = -1;
  }
  /**
   * Toggles the membership of the given layer.
   *
   * @param {number} layer - The layer to toggle.
   */
  toggle(e) {
    this.mask ^= 1 << e | 0;
  }
  /**
   * Removes membership of the given layer.
   *
   * @param {number} layer - The layer to enable.
   */
  disable(e) {
    this.mask &= ~(1 << e | 0);
  }
  /**
   * Removes the membership from all layers.
   */
  disableAll() {
    this.mask = 0;
  }
  /**
   * Returns `true` if this and the given layers object have at least one
   * layer in common.
   *
   * @param {Layers} layers - The layers to test.
   * @return {boolean } Whether this and the given layers object have at least one layer in common or not.
   */
  test(e) {
    return (this.mask & e.mask) !== 0;
  }
  /**
   * Returns `true` if the given layer is enabled.
   *
   * @param {number} layer - The layer to test.
   * @return {boolean } Whether the given layer is enabled or not.
   */
  isEnabled(e) {
    return (this.mask & (1 << e | 0)) !== 0;
  }
}
let Hg = 0;
const Ou = /* @__PURE__ */ new N(), ls = /* @__PURE__ */ new Yi(), Xn = /* @__PURE__ */ new pt(), jr = /* @__PURE__ */ new N(), Js = /* @__PURE__ */ new N(), Vg = /* @__PURE__ */ new N(), kg = /* @__PURE__ */ new Yi(), Bu = /* @__PURE__ */ new N(1, 0, 0), zu = /* @__PURE__ */ new N(0, 1, 0), Hu = /* @__PURE__ */ new N(0, 0, 1), Vu = { type: "added" }, Gg = { type: "removed" }, cs = { type: "childadded", child: null }, Pa = { type: "childremoved", child: null };
class Tt extends Zi {
  /**
   * Constructs a new 3D object.
   */
  constructor() {
    super(), this.isObject3D = !0, Object.defineProperty(this, "id", { value: Hg++ }), this.uuid = Ir(), this.name = "", this.type = "Object3D", this.parent = null, this.children = [], this.up = Tt.DEFAULT_UP.clone();
    const e = new N(), t = new zn(), i = new Yi(), s = new N(1, 1, 1);
    function r() {
      i.setFromEuler(t, !1);
    }
    function o() {
      t.setFromQuaternion(i, void 0, !1);
    }
    t._onChange(r), i._onChange(o), Object.defineProperties(this, {
      /**
       * Represents the object's local position.
       *
       * @name Object3D#position
       * @type {Vector3}
       * @default (0,0,0)
       */
      position: {
        configurable: !0,
        enumerable: !0,
        value: e
      },
      /**
       * Represents the object's local rotation as Euler angles, in radians.
       *
       * @name Object3D#rotation
       * @type {Euler}
       * @default (0,0,0)
       */
      rotation: {
        configurable: !0,
        enumerable: !0,
        value: t
      },
      /**
       * Represents the object's local rotation as Quaternions.
       *
       * @name Object3D#quaternion
       * @type {Quaternion}
       */
      quaternion: {
        configurable: !0,
        enumerable: !0,
        value: i
      },
      /**
       * Represents the object's local scale.
       *
       * @name Object3D#scale
       * @type {Vector3}
       * @default (1,1,1)
       */
      scale: {
        configurable: !0,
        enumerable: !0,
        value: s
      },
      /**
       * Represents the object's model-view matrix.
       *
       * @name Object3D#modelViewMatrix
       * @type {Matrix4}
       */
      modelViewMatrix: {
        value: new pt()
      },
      /**
       * Represents the object's normal matrix.
       *
       * @name Object3D#normalMatrix
       * @type {Matrix3}
       */
      normalMatrix: {
        value: new Ye()
      }
    }), this.matrix = new pt(), this.matrixWorld = new pt(), this.matrixAutoUpdate = Tt.DEFAULT_MATRIX_AUTO_UPDATE, this.matrixWorldAutoUpdate = Tt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE, this.matrixWorldNeedsUpdate = !1, this.layers = new pd(), this.visible = !0, this.castShadow = !1, this.receiveShadow = !1, this.frustumCulled = !0, this.renderOrder = 0, this.animations = [], this.customDepthMaterial = void 0, this.customDistanceMaterial = void 0, this.userData = {};
  }
  /**
   * A callback that is executed immediately before a 3D object is rendered to a shadow map.
   *
   * @param {Renderer|WebGLRenderer} renderer - The renderer.
   * @param {Object3D} object - The 3D object.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {Camera} shadowCamera - The shadow camera.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Material} depthMaterial - The depth material.
   * @param {Object} group - The geometry group data.
   */
  onBeforeShadow() {
  }
  /**
   * A callback that is executed immediately after a 3D object is rendered to a shadow map.
   *
   * @param {Renderer|WebGLRenderer} renderer - The renderer.
   * @param {Object3D} object - The 3D object.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {Camera} shadowCamera - The shadow camera.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Material} depthMaterial - The depth material.
   * @param {Object} group - The geometry group data.
   */
  onAfterShadow() {
  }
  /**
   * A callback that is executed immediately before a 3D object is rendered.
   *
   * @param {Renderer|WebGLRenderer} renderer - The renderer.
   * @param {Object3D} object - The 3D object.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Material} material - The 3D object's material.
   * @param {Object} group - The geometry group data.
   */
  onBeforeRender() {
  }
  /**
   * A callback that is executed immediately after a 3D object is rendered.
   *
   * @param {Renderer|WebGLRenderer} renderer - The renderer.
   * @param {Object3D} object - The 3D object.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Material} material - The 3D object's material.
   * @param {Object} group - The geometry group data.
   */
  onAfterRender() {
  }
  /**
   * Applies the given transformation matrix to the object and updates the object's position,
   * rotation and scale.
   *
   * @param {Matrix4} matrix - The transformation matrix.
   */
  applyMatrix4(e) {
    this.matrixAutoUpdate && this.updateMatrix(), this.matrix.premultiply(e), this.matrix.decompose(this.position, this.quaternion, this.scale);
  }
  /**
   * Applies a rotation represented by given the quaternion to the 3D object.
   *
   * @param {Quaternion} q - The quaternion.
   * @return {Object3D} A reference to this instance.
   */
  applyQuaternion(e) {
    return this.quaternion.premultiply(e), this;
  }
  /**
   * Sets the given rotation represented as an axis/angle couple to the 3D object.
   *
   * @param {Vector3} axis - The (normalized) axis vector.
   * @param {number} angle - The angle in radians.
   */
  setRotationFromAxisAngle(e, t) {
    this.quaternion.setFromAxisAngle(e, t);
  }
  /**
   * Sets the given rotation represented as Euler angles to the 3D object.
   *
   * @param {Euler} euler - The Euler angles.
   */
  setRotationFromEuler(e) {
    this.quaternion.setFromEuler(e, !0);
  }
  /**
   * Sets the given rotation represented as rotation matrix to the 3D object.
   *
   * @param {Matrix4} m - Although a 4x4 matrix is expected, the upper 3x3 portion must be
   * a pure rotation matrix (i.e, unscaled).
   */
  setRotationFromMatrix(e) {
    this.quaternion.setFromRotationMatrix(e);
  }
  /**
   * Sets the given rotation represented as a Quaternion to the 3D object.
   *
   * @param {Quaternion} q - The Quaternion
   */
  setRotationFromQuaternion(e) {
    this.quaternion.copy(e);
  }
  /**
   * Rotates the 3D object along an axis in local space.
   *
   * @param {Vector3} axis - The (normalized) axis vector.
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateOnAxis(e, t) {
    return ls.setFromAxisAngle(e, t), this.quaternion.multiply(ls), this;
  }
  /**
   * Rotates the 3D object along an axis in world space.
   *
   * @param {Vector3} axis - The (normalized) axis vector.
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateOnWorldAxis(e, t) {
    return ls.setFromAxisAngle(e, t), this.quaternion.premultiply(ls), this;
  }
  /**
   * Rotates the 3D object around its X axis in local space.
   *
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateX(e) {
    return this.rotateOnAxis(Bu, e);
  }
  /**
   * Rotates the 3D object around its Y axis in local space.
   *
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateY(e) {
    return this.rotateOnAxis(zu, e);
  }
  /**
   * Rotates the 3D object around its Z axis in local space.
   *
   * @param {number} angle - The angle in radians.
   * @return {Object3D} A reference to this instance.
   */
  rotateZ(e) {
    return this.rotateOnAxis(Hu, e);
  }
  /**
   * Translate the 3D object by a distance along the given axis in local space.
   *
   * @param {Vector3} axis - The (normalized) axis vector.
   * @param {number} distance - The distance in world units.
   * @return {Object3D} A reference to this instance.
   */
  translateOnAxis(e, t) {
    return Ou.copy(e).applyQuaternion(this.quaternion), this.position.add(Ou.multiplyScalar(t)), this;
  }
  /**
   * Translate the 3D object by a distance along its X-axis in local space.
   *
   * @param {number} distance - The distance in world units.
   * @return {Object3D} A reference to this instance.
   */
  translateX(e) {
    return this.translateOnAxis(Bu, e);
  }
  /**
   * Translate the 3D object by a distance along its Y-axis in local space.
   *
   * @param {number} distance - The distance in world units.
   * @return {Object3D} A reference to this instance.
   */
  translateY(e) {
    return this.translateOnAxis(zu, e);
  }
  /**
   * Translate the 3D object by a distance along its Z-axis in local space.
   *
   * @param {number} distance - The distance in world units.
   * @return {Object3D} A reference to this instance.
   */
  translateZ(e) {
    return this.translateOnAxis(Hu, e);
  }
  /**
   * Converts the given vector from this 3D object's local space to world space.
   *
   * @param {Vector3} vector - The vector to convert.
   * @return {Vector3} The converted vector.
   */
  localToWorld(e) {
    return this.updateWorldMatrix(!0, !1), e.applyMatrix4(this.matrixWorld);
  }
  /**
   * Converts the given vector from this 3D object's word space to local space.
   *
   * @param {Vector3} vector - The vector to convert.
   * @return {Vector3} The converted vector.
   */
  worldToLocal(e) {
    return this.updateWorldMatrix(!0, !1), e.applyMatrix4(Xn.copy(this.matrixWorld).invert());
  }
  /**
   * Rotates the object to face a point in world space.
   *
   * This method does not support objects having non-uniformly-scaled parent(s).
   *
   * @param {number|Vector3} x - The x coordinate in world space. Alternatively, a vector representing a position in world space
   * @param {number} [y] - The y coordinate in world space.
   * @param {number} [z] - The z coordinate in world space.
   */
  lookAt(e, t, i) {
    e.isVector3 ? jr.copy(e) : jr.set(e, t, i);
    const s = this.parent;
    this.updateWorldMatrix(!0, !1), Js.setFromMatrixPosition(this.matrixWorld), this.isCamera || this.isLight ? Xn.lookAt(Js, jr, this.up) : Xn.lookAt(jr, Js, this.up), this.quaternion.setFromRotationMatrix(Xn), s && (Xn.extractRotation(s.matrixWorld), ls.setFromRotationMatrix(Xn), this.quaternion.premultiply(ls.invert()));
  }
  /**
   * Adds the given 3D object as a child to this 3D object. An arbitrary number of
   * objects may be added. Any current parent on an object passed in here will be
   * removed, since an object can have at most one parent.
   *
   * @fires Object3D#added
   * @fires Object3D#childadded
   * @param {Object3D} object - The 3D object to add.
   * @return {Object3D} A reference to this instance.
   */
  add(e) {
    if (arguments.length > 1) {
      for (let t = 0; t < arguments.length; t++)
        this.add(arguments[t]);
      return this;
    }
    return e === this ? (console.error("THREE.Object3D.add: object can't be added as a child of itself.", e), this) : (e && e.isObject3D ? (e.removeFromParent(), e.parent = this, this.children.push(e), e.dispatchEvent(Vu), cs.child = e, this.dispatchEvent(cs), cs.child = null) : console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.", e), this);
  }
  /**
   * Removes the given 3D object as child from this 3D object.
   * An arbitrary number of objects may be removed.
   *
   * @fires Object3D#removed
   * @fires Object3D#childremoved
   * @param {Object3D} object - The 3D object to remove.
   * @return {Object3D} A reference to this instance.
   */
  remove(e) {
    if (arguments.length > 1) {
      for (let i = 0; i < arguments.length; i++)
        this.remove(arguments[i]);
      return this;
    }
    const t = this.children.indexOf(e);
    return t !== -1 && (e.parent = null, this.children.splice(t, 1), e.dispatchEvent(Gg), Pa.child = e, this.dispatchEvent(Pa), Pa.child = null), this;
  }
  /**
   * Removes this 3D object from its current parent.
   *
   * @fires Object3D#removed
   * @fires Object3D#childremoved
   * @return {Object3D} A reference to this instance.
   */
  removeFromParent() {
    const e = this.parent;
    return e !== null && e.remove(this), this;
  }
  /**
   * Removes all child objects.
   *
   * @fires Object3D#removed
   * @fires Object3D#childremoved
   * @return {Object3D} A reference to this instance.
   */
  clear() {
    return this.remove(...this.children);
  }
  /**
   * Adds the given 3D object as a child of this 3D object, while maintaining the object's world
   * transform. This method does not support scene graphs having non-uniformly-scaled nodes(s).
   *
   * @fires Object3D#added
   * @fires Object3D#childadded
   * @param {Object3D} object - The 3D object to attach.
   * @return {Object3D} A reference to this instance.
   */
  attach(e) {
    return this.updateWorldMatrix(!0, !1), Xn.copy(this.matrixWorld).invert(), e.parent !== null && (e.parent.updateWorldMatrix(!0, !1), Xn.multiply(e.parent.matrixWorld)), e.applyMatrix4(Xn), e.removeFromParent(), e.parent = this, this.children.push(e), e.updateWorldMatrix(!1, !0), e.dispatchEvent(Vu), cs.child = e, this.dispatchEvent(cs), cs.child = null, this;
  }
  /**
   * Searches through the 3D object and its children, starting with the 3D object
   * itself, and returns the first with a matching ID.
   *
   * @param {number} id - The id.
   * @return {Object3D|undefined} The found 3D object. Returns `undefined` if no 3D object has been found.
   */
  getObjectById(e) {
    return this.getObjectByProperty("id", e);
  }
  /**
   * Searches through the 3D object and its children, starting with the 3D object
   * itself, and returns the first with a matching name.
   *
   * @param {string} name - The name.
   * @return {Object3D|undefined} The found 3D object. Returns `undefined` if no 3D object has been found.
   */
  getObjectByName(e) {
    return this.getObjectByProperty("name", e);
  }
  /**
   * Searches through the 3D object and its children, starting with the 3D object
   * itself, and returns the first with a matching property value.
   *
   * @param {string} name - The name of the property.
   * @param {any} value - The value.
   * @return {Object3D|undefined} The found 3D object. Returns `undefined` if no 3D object has been found.
   */
  getObjectByProperty(e, t) {
    if (this[e] === t) return this;
    for (let i = 0, s = this.children.length; i < s; i++) {
      const o = this.children[i].getObjectByProperty(e, t);
      if (o !== void 0)
        return o;
    }
  }
  /**
   * Searches through the 3D object and its children, starting with the 3D object
   * itself, and returns all 3D objects with a matching property value.
   *
   * @param {string} name - The name of the property.
   * @param {any} value - The value.
   * @param {Array<Object3D>} result - The method stores the result in this array.
   * @return {Array<Object3D>} The found 3D objects.
   */
  getObjectsByProperty(e, t, i = []) {
    this[e] === t && i.push(this);
    const s = this.children;
    for (let r = 0, o = s.length; r < o; r++)
      s[r].getObjectsByProperty(e, t, i);
    return i;
  }
  /**
   * Returns a vector representing the position of the 3D object in world space.
   *
   * @param {Vector3} target - The target vector the result is stored to.
   * @return {Vector3} The 3D object's position in world space.
   */
  getWorldPosition(e) {
    return this.updateWorldMatrix(!0, !1), e.setFromMatrixPosition(this.matrixWorld);
  }
  /**
   * Returns a Quaternion representing the position of the 3D object in world space.
   *
   * @param {Quaternion} target - The target Quaternion the result is stored to.
   * @return {Quaternion} The 3D object's rotation in world space.
   */
  getWorldQuaternion(e) {
    return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(Js, e, Vg), e;
  }
  /**
   * Returns a vector representing the scale of the 3D object in world space.
   *
   * @param {Vector3} target - The target vector the result is stored to.
   * @return {Vector3} The 3D object's scale in world space.
   */
  getWorldScale(e) {
    return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(Js, kg, e), e;
  }
  /**
   * Returns a vector representing the ("look") direction of the 3D object in world space.
   *
   * @param {Vector3} target - The target vector the result is stored to.
   * @return {Vector3} The 3D object's direction in world space.
   */
  getWorldDirection(e) {
    this.updateWorldMatrix(!0, !1);
    const t = this.matrixWorld.elements;
    return e.set(t[8], t[9], t[10]).normalize();
  }
  /**
   * Abstract method to get intersections between a casted ray and this
   * 3D object. Renderable 3D objects such as {@link Mesh}, {@link Line} or {@link Points}
   * implement this method in order to use raycasting.
   *
   * @abstract
   * @param {Raycaster} raycaster - The raycaster.
   * @param {Array<Object>} intersects - An array holding the result of the method.
   */
  raycast() {
  }
  /**
   * Executes the callback on this 3D object and all descendants.
   *
   * Note: Modifying the scene graph inside the callback is discouraged.
   *
   * @param {Function} callback - A callback function that allows to process the current 3D object.
   */
  traverse(e) {
    e(this);
    const t = this.children;
    for (let i = 0, s = t.length; i < s; i++)
      t[i].traverse(e);
  }
  /**
   * Like {@link Object3D#traverse}, but the callback will only be executed for visible 3D objects.
   * Descendants of invisible 3D objects are not traversed.
   *
   * Note: Modifying the scene graph inside the callback is discouraged.
   *
   * @param {Function} callback - A callback function that allows to process the current 3D object.
   */
  traverseVisible(e) {
    if (this.visible === !1) return;
    e(this);
    const t = this.children;
    for (let i = 0, s = t.length; i < s; i++)
      t[i].traverseVisible(e);
  }
  /**
   * Like {@link Object3D#traverse}, but the callback will only be executed for all ancestors.
   *
   * Note: Modifying the scene graph inside the callback is discouraged.
   *
   * @param {Function} callback - A callback function that allows to process the current 3D object.
   */
  traverseAncestors(e) {
    const t = this.parent;
    t !== null && (e(t), t.traverseAncestors(e));
  }
  /**
   * Updates the transformation matrix in local space by computing it from the current
   * position, rotation and scale values.
   */
  updateMatrix() {
    this.matrix.compose(this.position, this.quaternion, this.scale), this.matrixWorldNeedsUpdate = !0;
  }
  /**
   * Updates the transformation matrix in world space of this 3D objects and its descendants.
   *
   * To ensure correct results, this method also recomputes the 3D object's transformation matrix in
   * local space. The computation of the local and world matrix can be controlled with the
   * {@link Object3D#matrixAutoUpdate} and {@link Object3D#matrixWorldAutoUpdate} flags which are both
   * `true` by default.  Set these flags to `false` if you need more control over the update matrix process.
   *
   * @param {boolean} [force=false] - When set to `true`, a recomputation of world matrices is forced even
   * when {@link Object3D#matrixWorldAutoUpdate} is set to `false`.
   */
  updateMatrixWorld(e) {
    this.matrixAutoUpdate && this.updateMatrix(), (this.matrixWorldNeedsUpdate || e) && (this.matrixWorldAutoUpdate === !0 && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)), this.matrixWorldNeedsUpdate = !1, e = !0);
    const t = this.children;
    for (let i = 0, s = t.length; i < s; i++)
      t[i].updateMatrixWorld(e);
  }
  /**
   * An alternative version of {@link Object3D#updateMatrixWorld} with more control over the
   * update of ancestor and descendant nodes.
   *
   * @param {boolean} [updateParents=false] Whether ancestor nodes should be updated or not.
   * @param {boolean} [updateChildren=false] Whether descendant nodes should be updated or not.
   */
  updateWorldMatrix(e, t) {
    const i = this.parent;
    if (e === !0 && i !== null && i.updateWorldMatrix(!0, !1), this.matrixAutoUpdate && this.updateMatrix(), this.matrixWorldAutoUpdate === !0 && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)), t === !0) {
      const s = this.children;
      for (let r = 0, o = s.length; r < o; r++)
        s[r].updateWorldMatrix(!1, !0);
    }
  }
  /**
   * Serializes the 3D object into JSON.
   *
   * @param {?(Object|string)} meta - An optional value holding meta information about the serialization.
   * @return {Object} A JSON object representing the serialized 3D object.
   * @see {@link ObjectLoader#parse}
   */
  toJSON(e) {
    const t = e === void 0 || typeof e == "string", i = {};
    t && (e = {
      geometries: {},
      materials: {},
      textures: {},
      images: {},
      shapes: {},
      skeletons: {},
      animations: {},
      nodes: {}
    }, i.metadata = {
      version: 4.7,
      type: "Object",
      generator: "Object3D.toJSON"
    });
    const s = {};
    s.uuid = this.uuid, s.type = this.type, this.name !== "" && (s.name = this.name), this.castShadow === !0 && (s.castShadow = !0), this.receiveShadow === !0 && (s.receiveShadow = !0), this.visible === !1 && (s.visible = !1), this.frustumCulled === !1 && (s.frustumCulled = !1), this.renderOrder !== 0 && (s.renderOrder = this.renderOrder), Object.keys(this.userData).length > 0 && (s.userData = this.userData), s.layers = this.layers.mask, s.matrix = this.matrix.toArray(), s.up = this.up.toArray(), this.matrixAutoUpdate === !1 && (s.matrixAutoUpdate = !1), this.isInstancedMesh && (s.type = "InstancedMesh", s.count = this.count, s.instanceMatrix = this.instanceMatrix.toJSON(), this.instanceColor !== null && (s.instanceColor = this.instanceColor.toJSON())), this.isBatchedMesh && (s.type = "BatchedMesh", s.perObjectFrustumCulled = this.perObjectFrustumCulled, s.sortObjects = this.sortObjects, s.drawRanges = this._drawRanges, s.reservedRanges = this._reservedRanges, s.geometryInfo = this._geometryInfo.map((a) => ({
      ...a,
      boundingBox: a.boundingBox ? a.boundingBox.toJSON() : void 0,
      boundingSphere: a.boundingSphere ? a.boundingSphere.toJSON() : void 0
    })), s.instanceInfo = this._instanceInfo.map((a) => ({ ...a })), s.availableInstanceIds = this._availableInstanceIds.slice(), s.availableGeometryIds = this._availableGeometryIds.slice(), s.nextIndexStart = this._nextIndexStart, s.nextVertexStart = this._nextVertexStart, s.geometryCount = this._geometryCount, s.maxInstanceCount = this._maxInstanceCount, s.maxVertexCount = this._maxVertexCount, s.maxIndexCount = this._maxIndexCount, s.geometryInitialized = this._geometryInitialized, s.matricesTexture = this._matricesTexture.toJSON(e), s.indirectTexture = this._indirectTexture.toJSON(e), this._colorsTexture !== null && (s.colorsTexture = this._colorsTexture.toJSON(e)), this.boundingSphere !== null && (s.boundingSphere = this.boundingSphere.toJSON()), this.boundingBox !== null && (s.boundingBox = this.boundingBox.toJSON()));
    function r(a, l) {
      return a[l.uuid] === void 0 && (a[l.uuid] = l.toJSON(e)), l.uuid;
    }
    if (this.isScene)
      this.background && (this.background.isColor ? s.background = this.background.toJSON() : this.background.isTexture && (s.background = this.background.toJSON(e).uuid)), this.environment && this.environment.isTexture && this.environment.isRenderTargetTexture !== !0 && (s.environment = this.environment.toJSON(e).uuid);
    else if (this.isMesh || this.isLine || this.isPoints) {
      s.geometry = r(e.geometries, this.geometry);
      const a = this.geometry.parameters;
      if (a !== void 0 && a.shapes !== void 0) {
        const l = a.shapes;
        if (Array.isArray(l))
          for (let c = 0, u = l.length; c < u; c++) {
            const h = l[c];
            r(e.shapes, h);
          }
        else
          r(e.shapes, l);
      }
    }
    if (this.isSkinnedMesh && (s.bindMode = this.bindMode, s.bindMatrix = this.bindMatrix.toArray(), this.skeleton !== void 0 && (r(e.skeletons, this.skeleton), s.skeleton = this.skeleton.uuid)), this.material !== void 0)
      if (Array.isArray(this.material)) {
        const a = [];
        for (let l = 0, c = this.material.length; l < c; l++)
          a.push(r(e.materials, this.material[l]));
        s.material = a;
      } else
        s.material = r(e.materials, this.material);
    if (this.children.length > 0) {
      s.children = [];
      for (let a = 0; a < this.children.length; a++)
        s.children.push(this.children[a].toJSON(e).object);
    }
    if (this.animations.length > 0) {
      s.animations = [];
      for (let a = 0; a < this.animations.length; a++) {
        const l = this.animations[a];
        s.animations.push(r(e.animations, l));
      }
    }
    if (t) {
      const a = o(e.geometries), l = o(e.materials), c = o(e.textures), u = o(e.images), h = o(e.shapes), f = o(e.skeletons), p = o(e.animations), v = o(e.nodes);
      a.length > 0 && (i.geometries = a), l.length > 0 && (i.materials = l), c.length > 0 && (i.textures = c), u.length > 0 && (i.images = u), h.length > 0 && (i.shapes = h), f.length > 0 && (i.skeletons = f), p.length > 0 && (i.animations = p), v.length > 0 && (i.nodes = v);
    }
    return i.object = s, i;
    function o(a) {
      const l = [];
      for (const c in a) {
        const u = a[c];
        delete u.metadata, l.push(u);
      }
      return l;
    }
  }
  /**
   * Returns a new 3D object with copied values from this instance.
   *
   * @param {boolean} [recursive=true] - When set to `true`, descendants of the 3D object are also cloned.
   * @return {Object3D} A clone of this instance.
   */
  clone(e) {
    return new this.constructor().copy(this, e);
  }
  /**
   * Copies the values of the given 3D object to this instance.
   *
   * @param {Object3D} source - The 3D object to copy.
   * @param {boolean} [recursive=true] - When set to `true`, descendants of the 3D object are cloned.
   * @return {Object3D} A reference to this instance.
   */
  copy(e, t = !0) {
    if (this.name = e.name, this.up.copy(e.up), this.position.copy(e.position), this.rotation.order = e.rotation.order, this.quaternion.copy(e.quaternion), this.scale.copy(e.scale), this.matrix.copy(e.matrix), this.matrixWorld.copy(e.matrixWorld), this.matrixAutoUpdate = e.matrixAutoUpdate, this.matrixWorldAutoUpdate = e.matrixWorldAutoUpdate, this.matrixWorldNeedsUpdate = e.matrixWorldNeedsUpdate, this.layers.mask = e.layers.mask, this.visible = e.visible, this.castShadow = e.castShadow, this.receiveShadow = e.receiveShadow, this.frustumCulled = e.frustumCulled, this.renderOrder = e.renderOrder, this.animations = e.animations.slice(), this.userData = JSON.parse(JSON.stringify(e.userData)), t === !0)
      for (let i = 0; i < e.children.length; i++) {
        const s = e.children[i];
        this.add(s.clone());
      }
    return this;
  }
}
Tt.DEFAULT_UP = /* @__PURE__ */ new N(0, 1, 0);
Tt.DEFAULT_MATRIX_AUTO_UPDATE = !0;
Tt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE = !0;
const vn = /* @__PURE__ */ new N(), Yn = /* @__PURE__ */ new N(), Da = /* @__PURE__ */ new N(), qn = /* @__PURE__ */ new N(), us = /* @__PURE__ */ new N(), hs = /* @__PURE__ */ new N(), ku = /* @__PURE__ */ new N(), La = /* @__PURE__ */ new N(), Ia = /* @__PURE__ */ new N(), Ua = /* @__PURE__ */ new N(), Na = /* @__PURE__ */ new lt(), Fa = /* @__PURE__ */ new lt(), Oa = /* @__PURE__ */ new lt();
class fn {
  /**
   * Constructs a new triangle.
   *
   * @param {Vector3} [a=(0,0,0)] - The first corner of the triangle.
   * @param {Vector3} [b=(0,0,0)] - The second corner of the triangle.
   * @param {Vector3} [c=(0,0,0)] - The third corner of the triangle.
   */
  constructor(e = new N(), t = new N(), i = new N()) {
    this.a = e, this.b = t, this.c = i;
  }
  /**
   * Computes the normal vector of a triangle.
   *
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The triangle's normal.
   */
  static getNormal(e, t, i, s) {
    s.subVectors(i, t), vn.subVectors(e, t), s.cross(vn);
    const r = s.lengthSq();
    return r > 0 ? s.multiplyScalar(1 / Math.sqrt(r)) : s.set(0, 0, 0);
  }
  /**
   * Computes a barycentric coordinates from the given vector.
   * Returns `null` if the triangle is degenerate.
   *
   * @param {Vector3} point - A point in 3D space.
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The barycentric coordinates for the given point
   */
  static getBarycoord(e, t, i, s, r) {
    vn.subVectors(s, t), Yn.subVectors(i, t), Da.subVectors(e, t);
    const o = vn.dot(vn), a = vn.dot(Yn), l = vn.dot(Da), c = Yn.dot(Yn), u = Yn.dot(Da), h = o * c - a * a;
    if (h === 0)
      return r.set(0, 0, 0), null;
    const f = 1 / h, p = (c * l - a * u) * f, v = (o * u - a * l) * f;
    return r.set(1 - p - v, v, p);
  }
  /**
   * Returns `true` if the given point, when projected onto the plane of the
   * triangle, lies within the triangle.
   *
   * @param {Vector3} point - The point in 3D space to test.
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @return {boolean} Whether the given point, when projected onto the plane of the
   * triangle, lies within the triangle or not.
   */
  static containsPoint(e, t, i, s) {
    return this.getBarycoord(e, t, i, s, qn) === null ? !1 : qn.x >= 0 && qn.y >= 0 && qn.x + qn.y <= 1;
  }
  /**
   * Computes the value barycentrically interpolated for the given point on the
   * triangle. Returns `null` if the triangle is degenerate.
   *
   * @param {Vector3} point - Position of interpolated point.
   * @param {Vector3} p1 - The first corner of the triangle.
   * @param {Vector3} p2 - The second corner of the triangle.
   * @param {Vector3} p3 - The third corner of the triangle.
   * @param {Vector3} v1 - Value to interpolate of first vertex.
   * @param {Vector3} v2 - Value to interpolate of second vertex.
   * @param {Vector3} v3 - Value to interpolate of third vertex.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The interpolated value.
   */
  static getInterpolation(e, t, i, s, r, o, a, l) {
    return this.getBarycoord(e, t, i, s, qn) === null ? (l.x = 0, l.y = 0, "z" in l && (l.z = 0), "w" in l && (l.w = 0), null) : (l.setScalar(0), l.addScaledVector(r, qn.x), l.addScaledVector(o, qn.y), l.addScaledVector(a, qn.z), l);
  }
  /**
   * Computes the value barycentrically interpolated for the given attribute and indices.
   *
   * @param {BufferAttribute} attr - The attribute to interpolate.
   * @param {number} i1 - Index of first vertex.
   * @param {number} i2 - Index of second vertex.
   * @param {number} i3 - Index of third vertex.
   * @param {Vector3} barycoord - The barycoordinate value to use to interpolate.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The interpolated attribute value.
   */
  static getInterpolatedAttribute(e, t, i, s, r, o) {
    return Na.setScalar(0), Fa.setScalar(0), Oa.setScalar(0), Na.fromBufferAttribute(e, t), Fa.fromBufferAttribute(e, i), Oa.fromBufferAttribute(e, s), o.setScalar(0), o.addScaledVector(Na, r.x), o.addScaledVector(Fa, r.y), o.addScaledVector(Oa, r.z), o;
  }
  /**
   * Returns `true` if the triangle is oriented towards the given direction.
   *
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @param {Vector3} direction - The (normalized) direction vector.
   * @return {boolean} Whether the triangle is oriented towards the given direction or not.
   */
  static isFrontFacing(e, t, i, s) {
    return vn.subVectors(i, t), Yn.subVectors(e, t), vn.cross(Yn).dot(s) < 0;
  }
  /**
   * Sets the triangle's vertices by copying the given values.
   *
   * @param {Vector3} a - The first corner of the triangle.
   * @param {Vector3} b - The second corner of the triangle.
   * @param {Vector3} c - The third corner of the triangle.
   * @return {Triangle} A reference to this triangle.
   */
  set(e, t, i) {
    return this.a.copy(e), this.b.copy(t), this.c.copy(i), this;
  }
  /**
   * Sets the triangle's vertices by copying the given array values.
   *
   * @param {Array<Vector3>} points - An array with 3D points.
   * @param {number} i0 - The array index representing the first corner of the triangle.
   * @param {number} i1 - The array index representing the second corner of the triangle.
   * @param {number} i2 - The array index representing the third corner of the triangle.
   * @return {Triangle} A reference to this triangle.
   */
  setFromPointsAndIndices(e, t, i, s) {
    return this.a.copy(e[t]), this.b.copy(e[i]), this.c.copy(e[s]), this;
  }
  /**
   * Sets the triangle's vertices by copying the given attribute values.
   *
   * @param {BufferAttribute} attribute - A buffer attribute with 3D points data.
   * @param {number} i0 - The attribute index representing the first corner of the triangle.
   * @param {number} i1 - The attribute index representing the second corner of the triangle.
   * @param {number} i2 - The attribute index representing the third corner of the triangle.
   * @return {Triangle} A reference to this triangle.
   */
  setFromAttributeAndIndices(e, t, i, s) {
    return this.a.fromBufferAttribute(e, t), this.b.fromBufferAttribute(e, i), this.c.fromBufferAttribute(e, s), this;
  }
  /**
   * Returns a new triangle with copied values from this instance.
   *
   * @return {Triangle} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given triangle to this instance.
   *
   * @param {Triangle} triangle - The triangle to copy.
   * @return {Triangle} A reference to this triangle.
   */
  copy(e) {
    return this.a.copy(e.a), this.b.copy(e.b), this.c.copy(e.c), this;
  }
  /**
   * Computes the area of the triangle.
   *
   * @return {number} The triangle's area.
   */
  getArea() {
    return vn.subVectors(this.c, this.b), Yn.subVectors(this.a, this.b), vn.cross(Yn).length() * 0.5;
  }
  /**
   * Computes the midpoint of the triangle.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The triangle's midpoint.
   */
  getMidpoint(e) {
    return e.addVectors(this.a, this.b).add(this.c).multiplyScalar(1 / 3);
  }
  /**
   * Computes the normal of the triangle.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The triangle's normal.
   */
  getNormal(e) {
    return fn.getNormal(this.a, this.b, this.c, e);
  }
  /**
   * Computes a plane the triangle lies within.
   *
   * @param {Plane} target - The target vector that is used to store the method's result.
   * @return {Plane} The plane the triangle lies within.
   */
  getPlane(e) {
    return e.setFromCoplanarPoints(this.a, this.b, this.c);
  }
  /**
   * Computes a barycentric coordinates from the given vector.
   * Returns `null` if the triangle is degenerate.
   *
   * @param {Vector3} point - A point in 3D space.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The barycentric coordinates for the given point
   */
  getBarycoord(e, t) {
    return fn.getBarycoord(e, this.a, this.b, this.c, t);
  }
  /**
   * Computes the value barycentrically interpolated for the given point on the
   * triangle. Returns `null` if the triangle is degenerate.
   *
   * @param {Vector3} point - Position of interpolated point.
   * @param {Vector3} v1 - Value to interpolate of first vertex.
   * @param {Vector3} v2 - Value to interpolate of second vertex.
   * @param {Vector3} v3 - Value to interpolate of third vertex.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The interpolated value.
   */
  getInterpolation(e, t, i, s, r) {
    return fn.getInterpolation(e, this.a, this.b, this.c, t, i, s, r);
  }
  /**
   * Returns `true` if the given point, when projected onto the plane of the
   * triangle, lies within the triangle.
   *
   * @param {Vector3} point - The point in 3D space to test.
   * @return {boolean} Whether the given point, when projected onto the plane of the
   * triangle, lies within the triangle or not.
   */
  containsPoint(e) {
    return fn.containsPoint(e, this.a, this.b, this.c);
  }
  /**
   * Returns `true` if the triangle is oriented towards the given direction.
   *
   * @param {Vector3} direction - The (normalized) direction vector.
   * @return {boolean} Whether the triangle is oriented towards the given direction or not.
   */
  isFrontFacing(e) {
    return fn.isFrontFacing(this.a, this.b, this.c, e);
  }
  /**
   * Returns `true` if this triangle intersects with the given box.
   *
   * @param {Box3} box - The box to intersect.
   * @return {boolean} Whether this triangle intersects with the given box or not.
   */
  intersectsBox(e) {
    return e.intersectsTriangle(this);
  }
  /**
   * Returns the closest point on the triangle to the given point.
   *
   * @param {Vector3} p - The point to compute the closest point for.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The closest point on the triangle.
   */
  closestPointToPoint(e, t) {
    const i = this.a, s = this.b, r = this.c;
    let o, a;
    us.subVectors(s, i), hs.subVectors(r, i), La.subVectors(e, i);
    const l = us.dot(La), c = hs.dot(La);
    if (l <= 0 && c <= 0)
      return t.copy(i);
    Ia.subVectors(e, s);
    const u = us.dot(Ia), h = hs.dot(Ia);
    if (u >= 0 && h <= u)
      return t.copy(s);
    const f = l * h - u * c;
    if (f <= 0 && l >= 0 && u <= 0)
      return o = l / (l - u), t.copy(i).addScaledVector(us, o);
    Ua.subVectors(e, r);
    const p = us.dot(Ua), v = hs.dot(Ua);
    if (v >= 0 && p <= v)
      return t.copy(r);
    const x = p * c - l * v;
    if (x <= 0 && c >= 0 && v <= 0)
      return a = c / (c - v), t.copy(i).addScaledVector(hs, a);
    const m = u * v - p * h;
    if (m <= 0 && h - u >= 0 && p - v >= 0)
      return ku.subVectors(r, s), a = (h - u) / (h - u + (p - v)), t.copy(s).addScaledVector(ku, a);
    const d = 1 / (m + x + f);
    return o = x * d, a = f * d, t.copy(i).addScaledVector(us, o).addScaledVector(hs, a);
  }
  /**
   * Returns `true` if this triangle is equal with the given one.
   *
   * @param {Triangle} triangle - The triangle to test for equality.
   * @return {boolean} Whether this triangle is equal with the given one.
   */
  equals(e) {
    return e.a.equals(this.a) && e.b.equals(this.b) && e.c.equals(this.c);
  }
}
const md = {
  aliceblue: 15792383,
  antiquewhite: 16444375,
  aqua: 65535,
  aquamarine: 8388564,
  azure: 15794175,
  beige: 16119260,
  bisque: 16770244,
  black: 0,
  blanchedalmond: 16772045,
  blue: 255,
  blueviolet: 9055202,
  brown: 10824234,
  burlywood: 14596231,
  cadetblue: 6266528,
  chartreuse: 8388352,
  chocolate: 13789470,
  coral: 16744272,
  cornflowerblue: 6591981,
  cornsilk: 16775388,
  crimson: 14423100,
  cyan: 65535,
  darkblue: 139,
  darkcyan: 35723,
  darkgoldenrod: 12092939,
  darkgray: 11119017,
  darkgreen: 25600,
  darkgrey: 11119017,
  darkkhaki: 12433259,
  darkmagenta: 9109643,
  darkolivegreen: 5597999,
  darkorange: 16747520,
  darkorchid: 10040012,
  darkred: 9109504,
  darksalmon: 15308410,
  darkseagreen: 9419919,
  darkslateblue: 4734347,
  darkslategray: 3100495,
  darkslategrey: 3100495,
  darkturquoise: 52945,
  darkviolet: 9699539,
  deeppink: 16716947,
  deepskyblue: 49151,
  dimgray: 6908265,
  dimgrey: 6908265,
  dodgerblue: 2003199,
  firebrick: 11674146,
  floralwhite: 16775920,
  forestgreen: 2263842,
  fuchsia: 16711935,
  gainsboro: 14474460,
  ghostwhite: 16316671,
  gold: 16766720,
  goldenrod: 14329120,
  gray: 8421504,
  green: 32768,
  greenyellow: 11403055,
  grey: 8421504,
  honeydew: 15794160,
  hotpink: 16738740,
  indianred: 13458524,
  indigo: 4915330,
  ivory: 16777200,
  khaki: 15787660,
  lavender: 15132410,
  lavenderblush: 16773365,
  lawngreen: 8190976,
  lemonchiffon: 16775885,
  lightblue: 11393254,
  lightcoral: 15761536,
  lightcyan: 14745599,
  lightgoldenrodyellow: 16448210,
  lightgray: 13882323,
  lightgreen: 9498256,
  lightgrey: 13882323,
  lightpink: 16758465,
  lightsalmon: 16752762,
  lightseagreen: 2142890,
  lightskyblue: 8900346,
  lightslategray: 7833753,
  lightslategrey: 7833753,
  lightsteelblue: 11584734,
  lightyellow: 16777184,
  lime: 65280,
  limegreen: 3329330,
  linen: 16445670,
  magenta: 16711935,
  maroon: 8388608,
  mediumaquamarine: 6737322,
  mediumblue: 205,
  mediumorchid: 12211667,
  mediumpurple: 9662683,
  mediumseagreen: 3978097,
  mediumslateblue: 8087790,
  mediumspringgreen: 64154,
  mediumturquoise: 4772300,
  mediumvioletred: 13047173,
  midnightblue: 1644912,
  mintcream: 16121850,
  mistyrose: 16770273,
  moccasin: 16770229,
  navajowhite: 16768685,
  navy: 128,
  oldlace: 16643558,
  olive: 8421376,
  olivedrab: 7048739,
  orange: 16753920,
  orangered: 16729344,
  orchid: 14315734,
  palegoldenrod: 15657130,
  palegreen: 10025880,
  paleturquoise: 11529966,
  palevioletred: 14381203,
  papayawhip: 16773077,
  peachpuff: 16767673,
  peru: 13468991,
  pink: 16761035,
  plum: 14524637,
  powderblue: 11591910,
  purple: 8388736,
  rebeccapurple: 6697881,
  red: 16711680,
  rosybrown: 12357519,
  royalblue: 4286945,
  saddlebrown: 9127187,
  salmon: 16416882,
  sandybrown: 16032864,
  seagreen: 3050327,
  seashell: 16774638,
  sienna: 10506797,
  silver: 12632256,
  skyblue: 8900331,
  slateblue: 6970061,
  slategray: 7372944,
  slategrey: 7372944,
  snow: 16775930,
  springgreen: 65407,
  steelblue: 4620980,
  tan: 13808780,
  teal: 32896,
  thistle: 14204888,
  tomato: 16737095,
  turquoise: 4251856,
  violet: 15631086,
  wheat: 16113331,
  white: 16777215,
  whitesmoke: 16119285,
  yellow: 16776960,
  yellowgreen: 10145074
}, hi = { h: 0, s: 0, l: 0 }, Kr = { h: 0, s: 0, l: 0 };
function Ba(n, e, t) {
  return t < 0 && (t += 1), t > 1 && (t -= 1), t < 1 / 6 ? n + (e - n) * 6 * t : t < 1 / 2 ? e : t < 2 / 3 ? n + (e - n) * 6 * (2 / 3 - t) : n;
}
class We {
  /**
   * Constructs a new color.
   *
   * Note that standard method of specifying color in three.js is with a hexadecimal triplet,
   * and that method is used throughout the rest of the documentation.
   *
   * @param {(number|string|Color)} [r] - The red component of the color. If `g` and `b` are
   * not provided, it can be hexadecimal triplet, a CSS-style string or another `Color` instance.
   * @param {number} [g] - The green component.
   * @param {number} [b] - The blue component.
   */
  constructor(e, t, i) {
    return this.isColor = !0, this.r = 1, this.g = 1, this.b = 1, this.set(e, t, i);
  }
  /**
   * Sets the colors's components from the given values.
   *
   * @param {(number|string|Color)} [r] - The red component of the color. If `g` and `b` are
   * not provided, it can be hexadecimal triplet, a CSS-style string or another `Color` instance.
   * @param {number} [g] - The green component.
   * @param {number} [b] - The blue component.
   * @return {Color} A reference to this color.
   */
  set(e, t, i) {
    if (t === void 0 && i === void 0) {
      const s = e;
      s && s.isColor ? this.copy(s) : typeof s == "number" ? this.setHex(s) : typeof s == "string" && this.setStyle(s);
    } else
      this.setRGB(e, t, i);
    return this;
  }
  /**
   * Sets the colors's components to the given scalar value.
   *
   * @param {number} scalar - The scalar value.
   * @return {Color} A reference to this color.
   */
  setScalar(e) {
    return this.r = e, this.g = e, this.b = e, this;
  }
  /**
   * Sets this color from a hexadecimal value.
   *
   * @param {number} hex - The hexadecimal value.
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setHex(e, t = tn) {
    return e = Math.floor(e), this.r = (e >> 16 & 255) / 255, this.g = (e >> 8 & 255) / 255, this.b = (e & 255) / 255, Qe.colorSpaceToWorking(this, t), this;
  }
  /**
   * Sets this color from RGB values.
   *
   * @param {number} r - Red channel value between `0.0` and `1.0`.
   * @param {number} g - Green channel value between `0.0` and `1.0`.
   * @param {number} b - Blue channel value between `0.0` and `1.0`.
   * @param {string} [colorSpace=ColorManagement.workingColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setRGB(e, t, i, s = Qe.workingColorSpace) {
    return this.r = e, this.g = t, this.b = i, Qe.colorSpaceToWorking(this, s), this;
  }
  /**
   * Sets this color from RGB values.
   *
   * @param {number} h - Hue value between `0.0` and `1.0`.
   * @param {number} s - Saturation value between `0.0` and `1.0`.
   * @param {number} l - Lightness value between `0.0` and `1.0`.
   * @param {string} [colorSpace=ColorManagement.workingColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setHSL(e, t, i, s = Qe.workingColorSpace) {
    if (e = wg(e, 1), t = je(t, 0, 1), i = je(i, 0, 1), t === 0)
      this.r = this.g = this.b = i;
    else {
      const r = i <= 0.5 ? i * (1 + t) : i + t - i * t, o = 2 * i - r;
      this.r = Ba(o, r, e + 1 / 3), this.g = Ba(o, r, e), this.b = Ba(o, r, e - 1 / 3);
    }
    return Qe.colorSpaceToWorking(this, s), this;
  }
  /**
   * Sets this color from a CSS-style string. For example, `rgb(250, 0,0)`,
   * `rgb(100%, 0%, 0%)`, `hsl(0, 100%, 50%)`, `#ff0000`, `#f00`, or `red` ( or
   * any [X11 color name]{@link https://en.wikipedia.org/wiki/X11_color_names#Color_name_chart} -
   * all 140 color names are supported).
   *
   * @param {string} style - Color as a CSS-style string.
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setStyle(e, t = tn) {
    function i(r) {
      r !== void 0 && parseFloat(r) < 1 && console.warn("THREE.Color: Alpha component of " + e + " will be ignored.");
    }
    let s;
    if (s = /^(\w+)\(([^\)]*)\)/.exec(e)) {
      let r;
      const o = s[1], a = s[2];
      switch (o) {
        case "rgb":
        case "rgba":
          if (r = /^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return i(r[4]), this.setRGB(
              Math.min(255, parseInt(r[1], 10)) / 255,
              Math.min(255, parseInt(r[2], 10)) / 255,
              Math.min(255, parseInt(r[3], 10)) / 255,
              t
            );
          if (r = /^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return i(r[4]), this.setRGB(
              Math.min(100, parseInt(r[1], 10)) / 100,
              Math.min(100, parseInt(r[2], 10)) / 100,
              Math.min(100, parseInt(r[3], 10)) / 100,
              t
            );
          break;
        case "hsl":
        case "hsla":
          if (r = /^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return i(r[4]), this.setHSL(
              parseFloat(r[1]) / 360,
              parseFloat(r[2]) / 100,
              parseFloat(r[3]) / 100,
              t
            );
          break;
        default:
          console.warn("THREE.Color: Unknown color model " + e);
      }
    } else if (s = /^\#([A-Fa-f\d]+)$/.exec(e)) {
      const r = s[1], o = r.length;
      if (o === 3)
        return this.setRGB(
          parseInt(r.charAt(0), 16) / 15,
          parseInt(r.charAt(1), 16) / 15,
          parseInt(r.charAt(2), 16) / 15,
          t
        );
      if (o === 6)
        return this.setHex(parseInt(r, 16), t);
      console.warn("THREE.Color: Invalid hex color " + e);
    } else if (e && e.length > 0)
      return this.setColorName(e, t);
    return this;
  }
  /**
   * Sets this color from a color name. Faster than {@link Color#setStyle} if
   * you don't need the other CSS-style formats.
   *
   * For convenience, the list of names is exposed in `Color.NAMES` as a hash.
   * ```js
   * Color.NAMES.aliceblue // returns 0xF0F8FF
   * ```
   *
   * @param {string} style - The color name.
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {Color} A reference to this color.
   */
  setColorName(e, t = tn) {
    const i = md[e.toLowerCase()];
    return i !== void 0 ? this.setHex(i, t) : console.warn("THREE.Color: Unknown color " + e), this;
  }
  /**
   * Returns a new color with copied values from this instance.
   *
   * @return {Color} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.r, this.g, this.b);
  }
  /**
   * Copies the values of the given color to this instance.
   *
   * @param {Color} color - The color to copy.
   * @return {Color} A reference to this color.
   */
  copy(e) {
    return this.r = e.r, this.g = e.g, this.b = e.b, this;
  }
  /**
   * Copies the given color into this color, and then converts this color from
   * `SRGBColorSpace` to `LinearSRGBColorSpace`.
   *
   * @param {Color} color - The color to copy/convert.
   * @return {Color} A reference to this color.
   */
  copySRGBToLinear(e) {
    return this.r = ti(e.r), this.g = ti(e.g), this.b = ti(e.b), this;
  }
  /**
   * Copies the given color into this color, and then converts this color from
   * `LinearSRGBColorSpace` to `SRGBColorSpace`.
   *
   * @param {Color} color - The color to copy/convert.
   * @return {Color} A reference to this color.
   */
  copyLinearToSRGB(e) {
    return this.r = Ls(e.r), this.g = Ls(e.g), this.b = Ls(e.b), this;
  }
  /**
   * Converts this color from `SRGBColorSpace` to `LinearSRGBColorSpace`.
   *
   * @return {Color} A reference to this color.
   */
  convertSRGBToLinear() {
    return this.copySRGBToLinear(this), this;
  }
  /**
   * Converts this color from `LinearSRGBColorSpace` to `SRGBColorSpace`.
   *
   * @return {Color} A reference to this color.
   */
  convertLinearToSRGB() {
    return this.copyLinearToSRGB(this), this;
  }
  /**
   * Returns the hexadecimal value of this color.
   *
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {number} The hexadecimal value.
   */
  getHex(e = tn) {
    return Qe.workingToColorSpace(Dt.copy(this), e), Math.round(je(Dt.r * 255, 0, 255)) * 65536 + Math.round(je(Dt.g * 255, 0, 255)) * 256 + Math.round(je(Dt.b * 255, 0, 255));
  }
  /**
   * Returns the hexadecimal value of this color as a string (for example, 'FFFFFF').
   *
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {string} The hexadecimal value as a string.
   */
  getHexString(e = tn) {
    return ("000000" + this.getHex(e).toString(16)).slice(-6);
  }
  /**
   * Converts the colors RGB values into the HSL format and stores them into the
   * given target object.
   *
   * @param {{h:number,s:number,l:number}} target - The target object that is used to store the method's result.
   * @param {string} [colorSpace=ColorManagement.workingColorSpace] - The color space.
   * @return {{h:number,s:number,l:number}} The HSL representation of this color.
   */
  getHSL(e, t = Qe.workingColorSpace) {
    Qe.workingToColorSpace(Dt.copy(this), t);
    const i = Dt.r, s = Dt.g, r = Dt.b, o = Math.max(i, s, r), a = Math.min(i, s, r);
    let l, c;
    const u = (a + o) / 2;
    if (a === o)
      l = 0, c = 0;
    else {
      const h = o - a;
      switch (c = u <= 0.5 ? h / (o + a) : h / (2 - o - a), o) {
        case i:
          l = (s - r) / h + (s < r ? 6 : 0);
          break;
        case s:
          l = (r - i) / h + 2;
          break;
        case r:
          l = (i - s) / h + 4;
          break;
      }
      l /= 6;
    }
    return e.h = l, e.s = c, e.l = u, e;
  }
  /**
   * Returns the RGB values of this color and stores them into the given target object.
   *
   * @param {Color} target - The target color that is used to store the method's result.
   * @param {string} [colorSpace=ColorManagement.workingColorSpace] - The color space.
   * @return {Color} The RGB representation of this color.
   */
  getRGB(e, t = Qe.workingColorSpace) {
    return Qe.workingToColorSpace(Dt.copy(this), t), e.r = Dt.r, e.g = Dt.g, e.b = Dt.b, e;
  }
  /**
   * Returns the value of this color as a CSS style string. Example: `rgb(255,0,0)`.
   *
   * @param {string} [colorSpace=SRGBColorSpace] - The color space.
   * @return {string} The CSS representation of this color.
   */
  getStyle(e = tn) {
    Qe.workingToColorSpace(Dt.copy(this), e);
    const t = Dt.r, i = Dt.g, s = Dt.b;
    return e !== tn ? `color(${e} ${t.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})` : `rgb(${Math.round(t * 255)},${Math.round(i * 255)},${Math.round(s * 255)})`;
  }
  /**
   * Adds the given HSL values to this color's values.
   * Internally, this converts the color's RGB values to HSL, adds HSL
   * and then converts the color back to RGB.
   *
   * @param {number} h - Hue value between `0.0` and `1.0`.
   * @param {number} s - Saturation value between `0.0` and `1.0`.
   * @param {number} l - Lightness value between `0.0` and `1.0`.
   * @return {Color} A reference to this color.
   */
  offsetHSL(e, t, i) {
    return this.getHSL(hi), this.setHSL(hi.h + e, hi.s + t, hi.l + i);
  }
  /**
   * Adds the RGB values of the given color to the RGB values of this color.
   *
   * @param {Color} color - The color to add.
   * @return {Color} A reference to this color.
   */
  add(e) {
    return this.r += e.r, this.g += e.g, this.b += e.b, this;
  }
  /**
   * Adds the RGB values of the given colors and stores the result in this instance.
   *
   * @param {Color} color1 - The first color.
   * @param {Color} color2 - The second color.
   * @return {Color} A reference to this color.
   */
  addColors(e, t) {
    return this.r = e.r + t.r, this.g = e.g + t.g, this.b = e.b + t.b, this;
  }
  /**
   * Adds the given scalar value to the RGB values of this color.
   *
   * @param {number} s - The scalar to add.
   * @return {Color} A reference to this color.
   */
  addScalar(e) {
    return this.r += e, this.g += e, this.b += e, this;
  }
  /**
   * Subtracts the RGB values of the given color from the RGB values of this color.
   *
   * @param {Color} color - The color to subtract.
   * @return {Color} A reference to this color.
   */
  sub(e) {
    return this.r = Math.max(0, this.r - e.r), this.g = Math.max(0, this.g - e.g), this.b = Math.max(0, this.b - e.b), this;
  }
  /**
   * Multiplies the RGB values of the given color with the RGB values of this color.
   *
   * @param {Color} color - The color to multiply.
   * @return {Color} A reference to this color.
   */
  multiply(e) {
    return this.r *= e.r, this.g *= e.g, this.b *= e.b, this;
  }
  /**
   * Multiplies the given scalar value with the RGB values of this color.
   *
   * @param {number} s - The scalar to multiply.
   * @return {Color} A reference to this color.
   */
  multiplyScalar(e) {
    return this.r *= e, this.g *= e, this.b *= e, this;
  }
  /**
   * Linearly interpolates this color's RGB values toward the RGB values of the
   * given color. The alpha argument can be thought of as the ratio between
   * the two colors, where `0.0` is this color and `1.0` is the first argument.
   *
   * @param {Color} color - The color to converge on.
   * @param {number} alpha - The interpolation factor in the closed interval `[0,1]`.
   * @return {Color} A reference to this color.
   */
  lerp(e, t) {
    return this.r += (e.r - this.r) * t, this.g += (e.g - this.g) * t, this.b += (e.b - this.b) * t, this;
  }
  /**
   * Linearly interpolates between the given colors and stores the result in this instance.
   * The alpha argument can be thought of as the ratio between the two colors, where `0.0`
   * is the first and `1.0` is the second color.
   *
   * @param {Color} color1 - The first color.
   * @param {Color} color2 - The second color.
   * @param {number} alpha - The interpolation factor in the closed interval `[0,1]`.
   * @return {Color} A reference to this color.
   */
  lerpColors(e, t, i) {
    return this.r = e.r + (t.r - e.r) * i, this.g = e.g + (t.g - e.g) * i, this.b = e.b + (t.b - e.b) * i, this;
  }
  /**
   * Linearly interpolates this color's HSL values toward the HSL values of the
   * given color. It differs from {@link Color#lerp} by not interpolating straight
   * from one color to the other, but instead going through all the hues in between
   * those two colors. The alpha argument can be thought of as the ratio between
   * the two colors, where 0.0 is this color and 1.0 is the first argument.
   *
   * @param {Color} color - The color to converge on.
   * @param {number} alpha - The interpolation factor in the closed interval `[0,1]`.
   * @return {Color} A reference to this color.
   */
  lerpHSL(e, t) {
    this.getHSL(hi), e.getHSL(Kr);
    const i = Ma(hi.h, Kr.h, t), s = Ma(hi.s, Kr.s, t), r = Ma(hi.l, Kr.l, t);
    return this.setHSL(i, s, r), this;
  }
  /**
   * Sets the color's RGB components from the given 3D vector.
   *
   * @param {Vector3} v - The vector to set.
   * @return {Color} A reference to this color.
   */
  setFromVector3(e) {
    return this.r = e.x, this.g = e.y, this.b = e.z, this;
  }
  /**
   * Transforms this color with the given 3x3 matrix.
   *
   * @param {Matrix3} m - The matrix.
   * @return {Color} A reference to this color.
   */
  applyMatrix3(e) {
    const t = this.r, i = this.g, s = this.b, r = e.elements;
    return this.r = r[0] * t + r[3] * i + r[6] * s, this.g = r[1] * t + r[4] * i + r[7] * s, this.b = r[2] * t + r[5] * i + r[8] * s, this;
  }
  /**
   * Returns `true` if this color is equal with the given one.
   *
   * @param {Color} c - The color to test for equality.
   * @return {boolean} Whether this bounding color is equal with the given one.
   */
  equals(e) {
    return e.r === this.r && e.g === this.g && e.b === this.b;
  }
  /**
   * Sets this color's RGB components from the given array.
   *
   * @param {Array<number>} array - An array holding the RGB values.
   * @param {number} [offset=0] - The offset into the array.
   * @return {Color} A reference to this color.
   */
  fromArray(e, t = 0) {
    return this.r = e[t], this.g = e[t + 1], this.b = e[t + 2], this;
  }
  /**
   * Writes the RGB components of this color to the given array. If no array is provided,
   * the method returns a new instance.
   *
   * @param {Array<number>} [array=[]] - The target array holding the color components.
   * @param {number} [offset=0] - Index of the first element in the array.
   * @return {Array<number>} The color components.
   */
  toArray(e = [], t = 0) {
    return e[t] = this.r, e[t + 1] = this.g, e[t + 2] = this.b, e;
  }
  /**
   * Sets the components of this color from the given buffer attribute.
   *
   * @param {BufferAttribute} attribute - The buffer attribute holding color data.
   * @param {number} index - The index into the attribute.
   * @return {Color} A reference to this color.
   */
  fromBufferAttribute(e, t) {
    return this.r = e.getX(t), this.g = e.getY(t), this.b = e.getZ(t), this;
  }
  /**
   * This methods defines the serialization result of this class. Returns the color
   * as a hexadecimal value.
   *
   * @return {number} The hexadecimal value.
   */
  toJSON() {
    return this.getHex();
  }
  *[Symbol.iterator]() {
    yield this.r, yield this.g, yield this.b;
  }
}
const Dt = /* @__PURE__ */ new We();
We.NAMES = md;
let Wg = 0;
class Ji extends Zi {
  /**
   * Constructs a new material.
   */
  constructor() {
    super(), this.isMaterial = !0, Object.defineProperty(this, "id", { value: Wg++ }), this.uuid = Ir(), this.name = "", this.type = "Material", this.blending = Ds, this.side = Si, this.vertexColors = !1, this.opacity = 1, this.transparent = !1, this.alphaHash = !1, this.blendSrc = fl, this.blendDst = dl, this.blendEquation = Bi, this.blendSrcAlpha = null, this.blendDstAlpha = null, this.blendEquationAlpha = null, this.blendColor = new We(0, 0, 0), this.blendAlpha = 0, this.depthFunc = Us, this.depthTest = !0, this.depthWrite = !0, this.stencilWriteMask = 255, this.stencilFunc = Ru, this.stencilRef = 0, this.stencilFuncMask = 255, this.stencilFail = ns, this.stencilZFail = ns, this.stencilZPass = ns, this.stencilWrite = !1, this.clippingPlanes = null, this.clipIntersection = !1, this.clipShadows = !1, this.shadowSide = null, this.colorWrite = !0, this.precision = null, this.polygonOffset = !1, this.polygonOffsetFactor = 0, this.polygonOffsetUnits = 0, this.dithering = !1, this.alphaToCoverage = !1, this.premultipliedAlpha = !1, this.forceSinglePass = !1, this.allowOverride = !0, this.visible = !0, this.toneMapped = !0, this.userData = {}, this.version = 0, this._alphaTest = 0;
  }
  /**
   * Sets the alpha value to be used when running an alpha test. The material
   * will not be rendered if the opacity is lower than this value.
   *
   * @type {number}
   * @readonly
   * @default 0
   */
  get alphaTest() {
    return this._alphaTest;
  }
  set alphaTest(e) {
    this._alphaTest > 0 != e > 0 && this.version++, this._alphaTest = e;
  }
  /**
   * An optional callback that is executed immediately before the material is used to render a 3D object.
   *
   * This method can only be used when rendering with {@link WebGLRenderer}.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   * @param {Scene} scene - The scene.
   * @param {Camera} camera - The camera that is used to render the scene.
   * @param {BufferGeometry} geometry - The 3D object's geometry.
   * @param {Object3D} object - The 3D object.
   * @param {Object} group - The geometry group data.
   */
  onBeforeRender() {
  }
  /**
   * An optional callback that is executed immediately before the shader
   * program is compiled. This function is called with the shader source code
   * as a parameter. Useful for the modification of built-in materials.
   *
   * This method can only be used when rendering with {@link WebGLRenderer}. The
   * recommended approach when customizing materials is to use `WebGPURenderer` with the new
   * Node Material system and [TSL]{@link https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language}.
   *
   * @param {{vertexShader:string,fragmentShader:string,uniforms:Object}} shaderobject - The object holds the uniforms and the vertex and fragment shader source.
   * @param {WebGLRenderer} renderer - A reference to the renderer.
   */
  onBeforeCompile() {
  }
  /**
   * In case {@link Material#onBeforeCompile} is used, this callback can be used to identify
   * values of settings used in `onBeforeCompile()`, so three.js can reuse a cached
   * shader or recompile the shader for this material as needed.
   *
   * This method can only be used when rendering with {@link WebGLRenderer}.
   *
   * @return {string} The custom program cache key.
   */
  customProgramCacheKey() {
    return this.onBeforeCompile.toString();
  }
  /**
   * This method can be used to set default values from parameter objects.
   * It is a generic implementation so it can be used with different types
   * of materials.
   *
   * @param {Object} [values] - The material values to set.
   */
  setValues(e) {
    if (e !== void 0)
      for (const t in e) {
        const i = e[t];
        if (i === void 0) {
          console.warn(`THREE.Material: parameter '${t}' has value of undefined.`);
          continue;
        }
        const s = this[t];
        if (s === void 0) {
          console.warn(`THREE.Material: '${t}' is not a property of THREE.${this.type}.`);
          continue;
        }
        s && s.isColor ? s.set(i) : s && s.isVector3 && i && i.isVector3 ? s.copy(i) : this[t] = i;
      }
  }
  /**
   * Serializes the material into JSON.
   *
   * @param {?(Object|string)} meta - An optional value holding meta information about the serialization.
   * @return {Object} A JSON object representing the serialized material.
   * @see {@link ObjectLoader#parse}
   */
  toJSON(e) {
    const t = e === void 0 || typeof e == "string";
    t && (e = {
      textures: {},
      images: {}
    });
    const i = {
      metadata: {
        version: 4.7,
        type: "Material",
        generator: "Material.toJSON"
      }
    };
    i.uuid = this.uuid, i.type = this.type, this.name !== "" && (i.name = this.name), this.color && this.color.isColor && (i.color = this.color.getHex()), this.roughness !== void 0 && (i.roughness = this.roughness), this.metalness !== void 0 && (i.metalness = this.metalness), this.sheen !== void 0 && (i.sheen = this.sheen), this.sheenColor && this.sheenColor.isColor && (i.sheenColor = this.sheenColor.getHex()), this.sheenRoughness !== void 0 && (i.sheenRoughness = this.sheenRoughness), this.emissive && this.emissive.isColor && (i.emissive = this.emissive.getHex()), this.emissiveIntensity !== void 0 && this.emissiveIntensity !== 1 && (i.emissiveIntensity = this.emissiveIntensity), this.specular && this.specular.isColor && (i.specular = this.specular.getHex()), this.specularIntensity !== void 0 && (i.specularIntensity = this.specularIntensity), this.specularColor && this.specularColor.isColor && (i.specularColor = this.specularColor.getHex()), this.shininess !== void 0 && (i.shininess = this.shininess), this.clearcoat !== void 0 && (i.clearcoat = this.clearcoat), this.clearcoatRoughness !== void 0 && (i.clearcoatRoughness = this.clearcoatRoughness), this.clearcoatMap && this.clearcoatMap.isTexture && (i.clearcoatMap = this.clearcoatMap.toJSON(e).uuid), this.clearcoatRoughnessMap && this.clearcoatRoughnessMap.isTexture && (i.clearcoatRoughnessMap = this.clearcoatRoughnessMap.toJSON(e).uuid), this.clearcoatNormalMap && this.clearcoatNormalMap.isTexture && (i.clearcoatNormalMap = this.clearcoatNormalMap.toJSON(e).uuid, i.clearcoatNormalScale = this.clearcoatNormalScale.toArray()), this.sheenColorMap && this.sheenColorMap.isTexture && (i.sheenColorMap = this.sheenColorMap.toJSON(e).uuid), this.sheenRoughnessMap && this.sheenRoughnessMap.isTexture && (i.sheenRoughnessMap = this.sheenRoughnessMap.toJSON(e).uuid), this.dispersion !== void 0 && (i.dispersion = this.dispersion), this.iridescence !== void 0 && (i.iridescence = this.iridescence), this.iridescenceIOR !== void 0 && (i.iridescenceIOR = this.iridescenceIOR), this.iridescenceThicknessRange !== void 0 && (i.iridescenceThicknessRange = this.iridescenceThicknessRange), this.iridescenceMap && this.iridescenceMap.isTexture && (i.iridescenceMap = this.iridescenceMap.toJSON(e).uuid), this.iridescenceThicknessMap && this.iridescenceThicknessMap.isTexture && (i.iridescenceThicknessMap = this.iridescenceThicknessMap.toJSON(e).uuid), this.anisotropy !== void 0 && (i.anisotropy = this.anisotropy), this.anisotropyRotation !== void 0 && (i.anisotropyRotation = this.anisotropyRotation), this.anisotropyMap && this.anisotropyMap.isTexture && (i.anisotropyMap = this.anisotropyMap.toJSON(e).uuid), this.map && this.map.isTexture && (i.map = this.map.toJSON(e).uuid), this.matcap && this.matcap.isTexture && (i.matcap = this.matcap.toJSON(e).uuid), this.alphaMap && this.alphaMap.isTexture && (i.alphaMap = this.alphaMap.toJSON(e).uuid), this.lightMap && this.lightMap.isTexture && (i.lightMap = this.lightMap.toJSON(e).uuid, i.lightMapIntensity = this.lightMapIntensity), this.aoMap && this.aoMap.isTexture && (i.aoMap = this.aoMap.toJSON(e).uuid, i.aoMapIntensity = this.aoMapIntensity), this.bumpMap && this.bumpMap.isTexture && (i.bumpMap = this.bumpMap.toJSON(e).uuid, i.bumpScale = this.bumpScale), this.normalMap && this.normalMap.isTexture && (i.normalMap = this.normalMap.toJSON(e).uuid, i.normalMapType = this.normalMapType, i.normalScale = this.normalScale.toArray()), this.displacementMap && this.displacementMap.isTexture && (i.displacementMap = this.displacementMap.toJSON(e).uuid, i.displacementScale = this.displacementScale, i.displacementBias = this.displacementBias), this.roughnessMap && this.roughnessMap.isTexture && (i.roughnessMap = this.roughnessMap.toJSON(e).uuid), this.metalnessMap && this.metalnessMap.isTexture && (i.metalnessMap = this.metalnessMap.toJSON(e).uuid), this.emissiveMap && this.emissiveMap.isTexture && (i.emissiveMap = this.emissiveMap.toJSON(e).uuid), this.specularMap && this.specularMap.isTexture && (i.specularMap = this.specularMap.toJSON(e).uuid), this.specularIntensityMap && this.specularIntensityMap.isTexture && (i.specularIntensityMap = this.specularIntensityMap.toJSON(e).uuid), this.specularColorMap && this.specularColorMap.isTexture && (i.specularColorMap = this.specularColorMap.toJSON(e).uuid), this.envMap && this.envMap.isTexture && (i.envMap = this.envMap.toJSON(e).uuid, this.combine !== void 0 && (i.combine = this.combine)), this.envMapRotation !== void 0 && (i.envMapRotation = this.envMapRotation.toArray()), this.envMapIntensity !== void 0 && (i.envMapIntensity = this.envMapIntensity), this.reflectivity !== void 0 && (i.reflectivity = this.reflectivity), this.refractionRatio !== void 0 && (i.refractionRatio = this.refractionRatio), this.gradientMap && this.gradientMap.isTexture && (i.gradientMap = this.gradientMap.toJSON(e).uuid), this.transmission !== void 0 && (i.transmission = this.transmission), this.transmissionMap && this.transmissionMap.isTexture && (i.transmissionMap = this.transmissionMap.toJSON(e).uuid), this.thickness !== void 0 && (i.thickness = this.thickness), this.thicknessMap && this.thicknessMap.isTexture && (i.thicknessMap = this.thicknessMap.toJSON(e).uuid), this.attenuationDistance !== void 0 && this.attenuationDistance !== 1 / 0 && (i.attenuationDistance = this.attenuationDistance), this.attenuationColor !== void 0 && (i.attenuationColor = this.attenuationColor.getHex()), this.size !== void 0 && (i.size = this.size), this.shadowSide !== null && (i.shadowSide = this.shadowSide), this.sizeAttenuation !== void 0 && (i.sizeAttenuation = this.sizeAttenuation), this.blending !== Ds && (i.blending = this.blending), this.side !== Si && (i.side = this.side), this.vertexColors === !0 && (i.vertexColors = !0), this.opacity < 1 && (i.opacity = this.opacity), this.transparent === !0 && (i.transparent = !0), this.blendSrc !== fl && (i.blendSrc = this.blendSrc), this.blendDst !== dl && (i.blendDst = this.blendDst), this.blendEquation !== Bi && (i.blendEquation = this.blendEquation), this.blendSrcAlpha !== null && (i.blendSrcAlpha = this.blendSrcAlpha), this.blendDstAlpha !== null && (i.blendDstAlpha = this.blendDstAlpha), this.blendEquationAlpha !== null && (i.blendEquationAlpha = this.blendEquationAlpha), this.blendColor && this.blendColor.isColor && (i.blendColor = this.blendColor.getHex()), this.blendAlpha !== 0 && (i.blendAlpha = this.blendAlpha), this.depthFunc !== Us && (i.depthFunc = this.depthFunc), this.depthTest === !1 && (i.depthTest = this.depthTest), this.depthWrite === !1 && (i.depthWrite = this.depthWrite), this.colorWrite === !1 && (i.colorWrite = this.colorWrite), this.stencilWriteMask !== 255 && (i.stencilWriteMask = this.stencilWriteMask), this.stencilFunc !== Ru && (i.stencilFunc = this.stencilFunc), this.stencilRef !== 0 && (i.stencilRef = this.stencilRef), this.stencilFuncMask !== 255 && (i.stencilFuncMask = this.stencilFuncMask), this.stencilFail !== ns && (i.stencilFail = this.stencilFail), this.stencilZFail !== ns && (i.stencilZFail = this.stencilZFail), this.stencilZPass !== ns && (i.stencilZPass = this.stencilZPass), this.stencilWrite === !0 && (i.stencilWrite = this.stencilWrite), this.rotation !== void 0 && this.rotation !== 0 && (i.rotation = this.rotation), this.polygonOffset === !0 && (i.polygonOffset = !0), this.polygonOffsetFactor !== 0 && (i.polygonOffsetFactor = this.polygonOffsetFactor), this.polygonOffsetUnits !== 0 && (i.polygonOffsetUnits = this.polygonOffsetUnits), this.linewidth !== void 0 && this.linewidth !== 1 && (i.linewidth = this.linewidth), this.dashSize !== void 0 && (i.dashSize = this.dashSize), this.gapSize !== void 0 && (i.gapSize = this.gapSize), this.scale !== void 0 && (i.scale = this.scale), this.dithering === !0 && (i.dithering = !0), this.alphaTest > 0 && (i.alphaTest = this.alphaTest), this.alphaHash === !0 && (i.alphaHash = !0), this.alphaToCoverage === !0 && (i.alphaToCoverage = !0), this.premultipliedAlpha === !0 && (i.premultipliedAlpha = !0), this.forceSinglePass === !0 && (i.forceSinglePass = !0), this.wireframe === !0 && (i.wireframe = !0), this.wireframeLinewidth > 1 && (i.wireframeLinewidth = this.wireframeLinewidth), this.wireframeLinecap !== "round" && (i.wireframeLinecap = this.wireframeLinecap), this.wireframeLinejoin !== "round" && (i.wireframeLinejoin = this.wireframeLinejoin), this.flatShading === !0 && (i.flatShading = !0), this.visible === !1 && (i.visible = !1), this.toneMapped === !1 && (i.toneMapped = !1), this.fog === !1 && (i.fog = !1), Object.keys(this.userData).length > 0 && (i.userData = this.userData);
    function s(r) {
      const o = [];
      for (const a in r) {
        const l = r[a];
        delete l.metadata, o.push(l);
      }
      return o;
    }
    if (t) {
      const r = s(e.textures), o = s(e.images);
      r.length > 0 && (i.textures = r), o.length > 0 && (i.images = o);
    }
    return i;
  }
  /**
   * Returns a new material with copied values from this instance.
   *
   * @return {Material} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given material to this instance.
   *
   * @param {Material} source - The material to copy.
   * @return {Material} A reference to this instance.
   */
  copy(e) {
    this.name = e.name, this.blending = e.blending, this.side = e.side, this.vertexColors = e.vertexColors, this.opacity = e.opacity, this.transparent = e.transparent, this.blendSrc = e.blendSrc, this.blendDst = e.blendDst, this.blendEquation = e.blendEquation, this.blendSrcAlpha = e.blendSrcAlpha, this.blendDstAlpha = e.blendDstAlpha, this.blendEquationAlpha = e.blendEquationAlpha, this.blendColor.copy(e.blendColor), this.blendAlpha = e.blendAlpha, this.depthFunc = e.depthFunc, this.depthTest = e.depthTest, this.depthWrite = e.depthWrite, this.stencilWriteMask = e.stencilWriteMask, this.stencilFunc = e.stencilFunc, this.stencilRef = e.stencilRef, this.stencilFuncMask = e.stencilFuncMask, this.stencilFail = e.stencilFail, this.stencilZFail = e.stencilZFail, this.stencilZPass = e.stencilZPass, this.stencilWrite = e.stencilWrite;
    const t = e.clippingPlanes;
    let i = null;
    if (t !== null) {
      const s = t.length;
      i = new Array(s);
      for (let r = 0; r !== s; ++r)
        i[r] = t[r].clone();
    }
    return this.clippingPlanes = i, this.clipIntersection = e.clipIntersection, this.clipShadows = e.clipShadows, this.shadowSide = e.shadowSide, this.colorWrite = e.colorWrite, this.precision = e.precision, this.polygonOffset = e.polygonOffset, this.polygonOffsetFactor = e.polygonOffsetFactor, this.polygonOffsetUnits = e.polygonOffsetUnits, this.dithering = e.dithering, this.alphaTest = e.alphaTest, this.alphaHash = e.alphaHash, this.alphaToCoverage = e.alphaToCoverage, this.premultipliedAlpha = e.premultipliedAlpha, this.forceSinglePass = e.forceSinglePass, this.visible = e.visible, this.toneMapped = e.toneMapped, this.userData = JSON.parse(JSON.stringify(e.userData)), this;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   *
   * @fires Material#dispose
   */
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
  /**
   * Setting this property to `true` indicates the engine the material
   * needs to be recompiled.
   *
   * @type {boolean}
   * @default false
   * @param {boolean} value
   */
  set needsUpdate(e) {
    e === !0 && this.version++;
  }
}
class Rn extends Ji {
  /**
   * Constructs a new mesh basic material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshBasicMaterial = !0, this.type = "MeshBasicMaterial", this.color = new We(16777215), this.map = null, this.lightMap = null, this.lightMapIntensity = 1, this.aoMap = null, this.aoMapIntensity = 1, this.specularMap = null, this.alphaMap = null, this.envMap = null, this.envMapRotation = new zn(), this.combine = Qf, this.reflectivity = 1, this.refractionRatio = 0.98, this.wireframe = !1, this.wireframeLinewidth = 1, this.wireframeLinecap = "round", this.wireframeLinejoin = "round", this.fog = !0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.color.copy(e.color), this.map = e.map, this.lightMap = e.lightMap, this.lightMapIntensity = e.lightMapIntensity, this.aoMap = e.aoMap, this.aoMapIntensity = e.aoMapIntensity, this.specularMap = e.specularMap, this.alphaMap = e.alphaMap, this.envMap = e.envMap, this.envMapRotation.copy(e.envMapRotation), this.combine = e.combine, this.reflectivity = e.reflectivity, this.refractionRatio = e.refractionRatio, this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this.wireframeLinecap = e.wireframeLinecap, this.wireframeLinejoin = e.wireframeLinejoin, this.fog = e.fog, this;
  }
}
const St = /* @__PURE__ */ new N(), $r = /* @__PURE__ */ new Ve();
let Xg = 0;
class En {
  /**
   * Constructs a new buffer attribute.
   *
   * @param {TypedArray} array - The array holding the attribute data.
   * @param {number} itemSize - The item size.
   * @param {boolean} [normalized=false] - Whether the data are normalized or not.
   */
  constructor(e, t, i = !1) {
    if (Array.isArray(e))
      throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");
    this.isBufferAttribute = !0, Object.defineProperty(this, "id", { value: Xg++ }), this.name = "", this.array = e, this.itemSize = t, this.count = e !== void 0 ? e.length / t : 0, this.normalized = i, this.usage = Cu, this.updateRanges = [], this.gpuType = ei, this.version = 0;
  }
  /**
   * A callback function that is executed after the renderer has transferred the attribute
   * array data to the GPU.
   */
  onUploadCallback() {
  }
  /**
   * Flag to indicate that this attribute has changed and should be re-sent to
   * the GPU. Set this to `true` when you modify the value of the array.
   *
   * @type {number}
   * @default false
   * @param {boolean} value
   */
  set needsUpdate(e) {
    e === !0 && this.version++;
  }
  /**
   * Sets the usage of this buffer attribute.
   *
   * @param {(StaticDrawUsage|DynamicDrawUsage|StreamDrawUsage|StaticReadUsage|DynamicReadUsage|StreamReadUsage|StaticCopyUsage|DynamicCopyUsage|StreamCopyUsage)} value - The usage to set.
   * @return {BufferAttribute} A reference to this buffer attribute.
   */
  setUsage(e) {
    return this.usage = e, this;
  }
  /**
   * Adds a range of data in the data array to be updated on the GPU.
   *
   * @param {number} start - Position at which to start update.
   * @param {number} count - The number of components to update.
   */
  addUpdateRange(e, t) {
    this.updateRanges.push({ start: e, count: t });
  }
  /**
   * Clears the update ranges.
   */
  clearUpdateRanges() {
    this.updateRanges.length = 0;
  }
  /**
   * Copies the values of the given buffer attribute to this instance.
   *
   * @param {BufferAttribute} source - The buffer attribute to copy.
   * @return {BufferAttribute} A reference to this instance.
   */
  copy(e) {
    return this.name = e.name, this.array = new e.array.constructor(e.array), this.itemSize = e.itemSize, this.count = e.count, this.normalized = e.normalized, this.usage = e.usage, this.gpuType = e.gpuType, this;
  }
  /**
   * Copies a vector from the given buffer attribute to this one. The start
   * and destination position in the attribute buffers are represented by the
   * given indices.
   *
   * @param {number} index1 - The destination index into this buffer attribute.
   * @param {BufferAttribute} attribute - The buffer attribute to copy from.
   * @param {number} index2 - The source index into the given buffer attribute.
   * @return {BufferAttribute} A reference to this instance.
   */
  copyAt(e, t, i) {
    e *= this.itemSize, i *= t.itemSize;
    for (let s = 0, r = this.itemSize; s < r; s++)
      this.array[e + s] = t.array[i + s];
    return this;
  }
  /**
   * Copies the given array data into this buffer attribute.
   *
   * @param {(TypedArray|Array)} array - The array to copy.
   * @return {BufferAttribute} A reference to this instance.
   */
  copyArray(e) {
    return this.array.set(e), this;
  }
  /**
   * Applies the given 3x3 matrix to the given attribute. Works with
   * item size `2` and `3`.
   *
   * @param {Matrix3} m - The matrix to apply.
   * @return {BufferAttribute} A reference to this instance.
   */
  applyMatrix3(e) {
    if (this.itemSize === 2)
      for (let t = 0, i = this.count; t < i; t++)
        $r.fromBufferAttribute(this, t), $r.applyMatrix3(e), this.setXY(t, $r.x, $r.y);
    else if (this.itemSize === 3)
      for (let t = 0, i = this.count; t < i; t++)
        St.fromBufferAttribute(this, t), St.applyMatrix3(e), this.setXYZ(t, St.x, St.y, St.z);
    return this;
  }
  /**
   * Applies the given 4x4 matrix to the given attribute. Only works with
   * item size `3`.
   *
   * @param {Matrix4} m - The matrix to apply.
   * @return {BufferAttribute} A reference to this instance.
   */
  applyMatrix4(e) {
    for (let t = 0, i = this.count; t < i; t++)
      St.fromBufferAttribute(this, t), St.applyMatrix4(e), this.setXYZ(t, St.x, St.y, St.z);
    return this;
  }
  /**
   * Applies the given 3x3 normal matrix to the given attribute. Only works with
   * item size `3`.
   *
   * @param {Matrix3} m - The normal matrix to apply.
   * @return {BufferAttribute} A reference to this instance.
   */
  applyNormalMatrix(e) {
    for (let t = 0, i = this.count; t < i; t++)
      St.fromBufferAttribute(this, t), St.applyNormalMatrix(e), this.setXYZ(t, St.x, St.y, St.z);
    return this;
  }
  /**
   * Applies the given 4x4 matrix to the given attribute. Only works with
   * item size `3` and with direction vectors.
   *
   * @param {Matrix4} m - The matrix to apply.
   * @return {BufferAttribute} A reference to this instance.
   */
  transformDirection(e) {
    for (let t = 0, i = this.count; t < i; t++)
      St.fromBufferAttribute(this, t), St.transformDirection(e), this.setXYZ(t, St.x, St.y, St.z);
    return this;
  }
  /**
   * Sets the given array data in the buffer attribute.
   *
   * @param {(TypedArray|Array)} value - The array data to set.
   * @param {number} [offset=0] - The offset in this buffer attribute's array.
   * @return {BufferAttribute} A reference to this instance.
   */
  set(e, t = 0) {
    return this.array.set(e, t), this;
  }
  /**
   * Returns the given component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} component - The component index.
   * @return {number} The returned value.
   */
  getComponent(e, t) {
    let i = this.array[e * this.itemSize + t];
    return this.normalized && (i = Ks(i, this.array)), i;
  }
  /**
   * Sets the given value to the given component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} component - The component index.
   * @param {number} value - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setComponent(e, t, i) {
    return this.normalized && (i = Yt(i, this.array)), this.array[e * this.itemSize + t] = i, this;
  }
  /**
   * Returns the x component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @return {number} The x component.
   */
  getX(e) {
    let t = this.array[e * this.itemSize];
    return this.normalized && (t = Ks(t, this.array)), t;
  }
  /**
   * Sets the x component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} x - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setX(e, t) {
    return this.normalized && (t = Yt(t, this.array)), this.array[e * this.itemSize] = t, this;
  }
  /**
   * Returns the y component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @return {number} The y component.
   */
  getY(e) {
    let t = this.array[e * this.itemSize + 1];
    return this.normalized && (t = Ks(t, this.array)), t;
  }
  /**
   * Sets the y component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} y - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setY(e, t) {
    return this.normalized && (t = Yt(t, this.array)), this.array[e * this.itemSize + 1] = t, this;
  }
  /**
   * Returns the z component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @return {number} The z component.
   */
  getZ(e) {
    let t = this.array[e * this.itemSize + 2];
    return this.normalized && (t = Ks(t, this.array)), t;
  }
  /**
   * Sets the z component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} z - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setZ(e, t) {
    return this.normalized && (t = Yt(t, this.array)), this.array[e * this.itemSize + 2] = t, this;
  }
  /**
   * Returns the w component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @return {number} The w component.
   */
  getW(e) {
    let t = this.array[e * this.itemSize + 3];
    return this.normalized && (t = Ks(t, this.array)), t;
  }
  /**
   * Sets the w component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} w - The value to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setW(e, t) {
    return this.normalized && (t = Yt(t, this.array)), this.array[e * this.itemSize + 3] = t, this;
  }
  /**
   * Sets the x and y component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} x - The value for the x component to set.
   * @param {number} y - The value for the y component to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setXY(e, t, i) {
    return e *= this.itemSize, this.normalized && (t = Yt(t, this.array), i = Yt(i, this.array)), this.array[e + 0] = t, this.array[e + 1] = i, this;
  }
  /**
   * Sets the x, y and z component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} x - The value for the x component to set.
   * @param {number} y - The value for the y component to set.
   * @param {number} z - The value for the z component to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setXYZ(e, t, i, s) {
    return e *= this.itemSize, this.normalized && (t = Yt(t, this.array), i = Yt(i, this.array), s = Yt(s, this.array)), this.array[e + 0] = t, this.array[e + 1] = i, this.array[e + 2] = s, this;
  }
  /**
   * Sets the x, y, z and w component of the vector at the given index.
   *
   * @param {number} index - The index into the buffer attribute.
   * @param {number} x - The value for the x component to set.
   * @param {number} y - The value for the y component to set.
   * @param {number} z - The value for the z component to set.
   * @param {number} w - The value for the w component to set.
   * @return {BufferAttribute} A reference to this instance.
   */
  setXYZW(e, t, i, s, r) {
    return e *= this.itemSize, this.normalized && (t = Yt(t, this.array), i = Yt(i, this.array), s = Yt(s, this.array), r = Yt(r, this.array)), this.array[e + 0] = t, this.array[e + 1] = i, this.array[e + 2] = s, this.array[e + 3] = r, this;
  }
  /**
   * Sets the given callback function that is executed after the Renderer has transferred
   * the attribute array data to the GPU. Can be used to perform clean-up operations after
   * the upload when attribute data are not needed anymore on the CPU side.
   *
   * @param {Function} callback - The `onUpload()` callback.
   * @return {BufferAttribute} A reference to this instance.
   */
  onUpload(e) {
    return this.onUploadCallback = e, this;
  }
  /**
   * Returns a new buffer attribute with copied values from this instance.
   *
   * @return {BufferAttribute} A clone of this instance.
   */
  clone() {
    return new this.constructor(this.array, this.itemSize).copy(this);
  }
  /**
   * Serializes the buffer attribute into JSON.
   *
   * @return {Object} A JSON object representing the serialized buffer attribute.
   */
  toJSON() {
    const e = {
      itemSize: this.itemSize,
      type: this.array.constructor.name,
      array: Array.from(this.array),
      normalized: this.normalized
    };
    return this.name !== "" && (e.name = this.name), this.usage !== Cu && (e.usage = this.usage), e;
  }
}
class _d extends En {
  /**
   * Constructs a new buffer attribute.
   *
   * @param {(Array<number>|Uint16Array)} array - The array holding the attribute data.
   * @param {number} itemSize - The item size.
   * @param {boolean} [normalized=false] - Whether the data are normalized or not.
   */
  constructor(e, t, i) {
    super(new Uint16Array(e), t, i);
  }
}
class gd extends En {
  /**
   * Constructs a new buffer attribute.
   *
   * @param {(Array<number>|Uint32Array)} array - The array holding the attribute data.
   * @param {number} itemSize - The item size.
   * @param {boolean} [normalized=false] - Whether the data are normalized or not.
   */
  constructor(e, t, i) {
    super(new Uint32Array(e), t, i);
  }
}
class mt extends En {
  /**
   * Constructs a new buffer attribute.
   *
   * @param {(Array<number>|Float32Array)} array - The array holding the attribute data.
   * @param {number} itemSize - The item size.
   * @param {boolean} [normalized=false] - Whether the data are normalized or not.
   */
  constructor(e, t, i) {
    super(new Float32Array(e), t, i);
  }
}
let Yg = 0;
const ln = /* @__PURE__ */ new pt(), za = /* @__PURE__ */ new Tt(), fs = /* @__PURE__ */ new N(), en = /* @__PURE__ */ new Ur(), Qs = /* @__PURE__ */ new Ur(), wt = /* @__PURE__ */ new N();
class Nt extends Zi {
  /**
   * Constructs a new geometry.
   */
  constructor() {
    super(), this.isBufferGeometry = !0, Object.defineProperty(this, "id", { value: Yg++ }), this.uuid = Ir(), this.name = "", this.type = "BufferGeometry", this.index = null, this.indirect = null, this.attributes = {}, this.morphAttributes = {}, this.morphTargetsRelative = !1, this.groups = [], this.boundingBox = null, this.boundingSphere = null, this.drawRange = { start: 0, count: 1 / 0 }, this.userData = {};
  }
  /**
   * Returns the index of this geometry.
   *
   * @return {?BufferAttribute} The index. Returns `null` if no index is defined.
   */
  getIndex() {
    return this.index;
  }
  /**
   * Sets the given index to this geometry.
   *
   * @param {Array<number>|BufferAttribute} index - The index to set.
   * @return {BufferGeometry} A reference to this instance.
   */
  setIndex(e) {
    return Array.isArray(e) ? this.index = new (fd(e) ? gd : _d)(e, 1) : this.index = e, this;
  }
  /**
   * Sets the given indirect attribute to this geometry.
   *
   * @param {BufferAttribute} indirect - The attribute holding indirect draw calls.
   * @return {BufferGeometry} A reference to this instance.
   */
  setIndirect(e) {
    return this.indirect = e, this;
  }
  /**
   * Returns the indirect attribute of this geometry.
   *
   * @return {?BufferAttribute} The indirect attribute. Returns `null` if no indirect attribute is defined.
   */
  getIndirect() {
    return this.indirect;
  }
  /**
   * Returns the buffer attribute for the given name.
   *
   * @param {string} name - The attribute name.
   * @return {BufferAttribute|InterleavedBufferAttribute|undefined} The buffer attribute.
   * Returns `undefined` if not attribute has been found.
   */
  getAttribute(e) {
    return this.attributes[e];
  }
  /**
   * Sets the given attribute for the given name.
   *
   * @param {string} name - The attribute name.
   * @param {BufferAttribute|InterleavedBufferAttribute} attribute - The attribute to set.
   * @return {BufferGeometry} A reference to this instance.
   */
  setAttribute(e, t) {
    return this.attributes[e] = t, this;
  }
  /**
   * Deletes the attribute for the given name.
   *
   * @param {string} name - The attribute name to delete.
   * @return {BufferGeometry} A reference to this instance.
   */
  deleteAttribute(e) {
    return delete this.attributes[e], this;
  }
  /**
   * Returns `true` if this geometry has an attribute for the given name.
   *
   * @param {string} name - The attribute name.
   * @return {boolean} Whether this geometry has an attribute for the given name or not.
   */
  hasAttribute(e) {
    return this.attributes[e] !== void 0;
  }
  /**
   * Adds a group to this geometry.
   *
   * @param {number} start - The first element in this draw call. That is the first
   * vertex for non-indexed geometry, otherwise the first triangle index.
   * @param {number} count - Specifies how many vertices (or indices) are part of this group.
   * @param {number} [materialIndex=0] - The material array index to use.
   */
  addGroup(e, t, i = 0) {
    this.groups.push({
      start: e,
      count: t,
      materialIndex: i
    });
  }
  /**
   * Clears all groups.
   */
  clearGroups() {
    this.groups = [];
  }
  /**
   * Sets the draw range for this geometry.
   *
   * @param {number} start - The first vertex for non-indexed geometry, otherwise the first triangle index.
   * @param {number} count - For non-indexed BufferGeometry, `count` is the number of vertices to render.
   * For indexed BufferGeometry, `count` is the number of indices to render.
   */
  setDrawRange(e, t) {
    this.drawRange.start = e, this.drawRange.count = t;
  }
  /**
   * Applies the given 4x4 transformation matrix to the geometry.
   *
   * @param {Matrix4} matrix - The matrix to apply.
   * @return {BufferGeometry} A reference to this instance.
   */
  applyMatrix4(e) {
    const t = this.attributes.position;
    t !== void 0 && (t.applyMatrix4(e), t.needsUpdate = !0);
    const i = this.attributes.normal;
    if (i !== void 0) {
      const r = new Ye().getNormalMatrix(e);
      i.applyNormalMatrix(r), i.needsUpdate = !0;
    }
    const s = this.attributes.tangent;
    return s !== void 0 && (s.transformDirection(e), s.needsUpdate = !0), this.boundingBox !== null && this.computeBoundingBox(), this.boundingSphere !== null && this.computeBoundingSphere(), this;
  }
  /**
   * Applies the rotation represented by the Quaternion to the geometry.
   *
   * @param {Quaternion} q - The Quaternion to apply.
   * @return {BufferGeometry} A reference to this instance.
   */
  applyQuaternion(e) {
    return ln.makeRotationFromQuaternion(e), this.applyMatrix4(ln), this;
  }
  /**
   * Rotates the geometry about the X axis. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#rotation} for typical
   * real-time mesh rotation.
   *
   * @param {number} angle - The angle in radians.
   * @return {BufferGeometry} A reference to this instance.
   */
  rotateX(e) {
    return ln.makeRotationX(e), this.applyMatrix4(ln), this;
  }
  /**
   * Rotates the geometry about the Y axis. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#rotation} for typical
   * real-time mesh rotation.
   *
   * @param {number} angle - The angle in radians.
   * @return {BufferGeometry} A reference to this instance.
   */
  rotateY(e) {
    return ln.makeRotationY(e), this.applyMatrix4(ln), this;
  }
  /**
   * Rotates the geometry about the Z axis. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#rotation} for typical
   * real-time mesh rotation.
   *
   * @param {number} angle - The angle in radians.
   * @return {BufferGeometry} A reference to this instance.
   */
  rotateZ(e) {
    return ln.makeRotationZ(e), this.applyMatrix4(ln), this;
  }
  /**
   * Translates the geometry. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#position} for typical
   * real-time mesh rotation.
   *
   * @param {number} x - The x offset.
   * @param {number} y - The y offset.
   * @param {number} z - The z offset.
   * @return {BufferGeometry} A reference to this instance.
   */
  translate(e, t, i) {
    return ln.makeTranslation(e, t, i), this.applyMatrix4(ln), this;
  }
  /**
   * Scales the geometry. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#scale} for typical
   * real-time mesh rotation.
   *
   * @param {number} x - The x scale.
   * @param {number} y - The y scale.
   * @param {number} z - The z scale.
   * @return {BufferGeometry} A reference to this instance.
   */
  scale(e, t, i) {
    return ln.makeScale(e, t, i), this.applyMatrix4(ln), this;
  }
  /**
   * Rotates the geometry to face a point in 3D space. This is typically done as a one time
   * operation, and not during a loop. Use {@link Object3D#lookAt} for typical
   * real-time mesh rotation.
   *
   * @param {Vector3} vector - The target point.
   * @return {BufferGeometry} A reference to this instance.
   */
  lookAt(e) {
    return za.lookAt(e), za.updateMatrix(), this.applyMatrix4(za.matrix), this;
  }
  /**
   * Center the geometry based on its bounding box.
   *
   * @return {BufferGeometry} A reference to this instance.
   */
  center() {
    return this.computeBoundingBox(), this.boundingBox.getCenter(fs).negate(), this.translate(fs.x, fs.y, fs.z), this;
  }
  /**
   * Defines a geometry by creating a `position` attribute based on the given array of points. The array
   * can hold 2D or 3D vectors. When using two-dimensional data, the `z` coordinate for all vertices is
   * set to `0`.
   *
   * If the method is used with an existing `position` attribute, the vertex data are overwritten with the
   * data from the array. The length of the array must match the vertex count.
   *
   * @param {Array<Vector2>|Array<Vector3>} points - The points.
   * @return {BufferGeometry} A reference to this instance.
   */
  setFromPoints(e) {
    const t = this.getAttribute("position");
    if (t === void 0) {
      const i = [];
      for (let s = 0, r = e.length; s < r; s++) {
        const o = e[s];
        i.push(o.x, o.y, o.z || 0);
      }
      this.setAttribute("position", new mt(i, 3));
    } else {
      const i = Math.min(e.length, t.count);
      for (let s = 0; s < i; s++) {
        const r = e[s];
        t.setXYZ(s, r.x, r.y, r.z || 0);
      }
      e.length > t.count && console.warn("THREE.BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."), t.needsUpdate = !0;
    }
    return this;
  }
  /**
   * Computes the bounding box of the geometry, and updates the `boundingBox` member.
   * The bounding box is not computed by the engine; it must be computed by your app.
   * You may need to recompute the bounding box if the geometry vertices are modified.
   */
  computeBoundingBox() {
    this.boundingBox === null && (this.boundingBox = new Ur());
    const e = this.attributes.position, t = this.morphAttributes.position;
    if (e && e.isGLBufferAttribute) {
      console.error("THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.", this), this.boundingBox.set(
        new N(-1 / 0, -1 / 0, -1 / 0),
        new N(1 / 0, 1 / 0, 1 / 0)
      );
      return;
    }
    if (e !== void 0) {
      if (this.boundingBox.setFromBufferAttribute(e), t)
        for (let i = 0, s = t.length; i < s; i++) {
          const r = t[i];
          en.setFromBufferAttribute(r), this.morphTargetsRelative ? (wt.addVectors(this.boundingBox.min, en.min), this.boundingBox.expandByPoint(wt), wt.addVectors(this.boundingBox.max, en.max), this.boundingBox.expandByPoint(wt)) : (this.boundingBox.expandByPoint(en.min), this.boundingBox.expandByPoint(en.max));
        }
    } else
      this.boundingBox.makeEmpty();
    (isNaN(this.boundingBox.min.x) || isNaN(this.boundingBox.min.y) || isNaN(this.boundingBox.min.z)) && console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.', this);
  }
  /**
   * Computes the bounding sphere of the geometry, and updates the `boundingSphere` member.
   * The engine automatically computes the bounding sphere when it is needed, e.g., for ray casting or view frustum culling.
   * You may need to recompute the bounding sphere if the geometry vertices are modified.
   */
  computeBoundingSphere() {
    this.boundingSphere === null && (this.boundingSphere = new Nr());
    const e = this.attributes.position, t = this.morphAttributes.position;
    if (e && e.isGLBufferAttribute) {
      console.error("THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.", this), this.boundingSphere.set(new N(), 1 / 0);
      return;
    }
    if (e) {
      const i = this.boundingSphere.center;
      if (en.setFromBufferAttribute(e), t)
        for (let r = 0, o = t.length; r < o; r++) {
          const a = t[r];
          Qs.setFromBufferAttribute(a), this.morphTargetsRelative ? (wt.addVectors(en.min, Qs.min), en.expandByPoint(wt), wt.addVectors(en.max, Qs.max), en.expandByPoint(wt)) : (en.expandByPoint(Qs.min), en.expandByPoint(Qs.max));
        }
      en.getCenter(i);
      let s = 0;
      for (let r = 0, o = e.count; r < o; r++)
        wt.fromBufferAttribute(e, r), s = Math.max(s, i.distanceToSquared(wt));
      if (t)
        for (let r = 0, o = t.length; r < o; r++) {
          const a = t[r], l = this.morphTargetsRelative;
          for (let c = 0, u = a.count; c < u; c++)
            wt.fromBufferAttribute(a, c), l && (fs.fromBufferAttribute(e, c), wt.add(fs)), s = Math.max(s, i.distanceToSquared(wt));
        }
      this.boundingSphere.radius = Math.sqrt(s), isNaN(this.boundingSphere.radius) && console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.', this);
    }
  }
  /**
   * Calculates and adds a tangent attribute to this geometry.
   *
   * The computation is only supported for indexed geometries and if position, normal, and uv attributes
   * are defined. When using a tangent space normal map, prefer the MikkTSpace algorithm provided by
   * {@link BufferGeometryUtils#computeMikkTSpaceTangents} instead.
   */
  computeTangents() {
    const e = this.index, t = this.attributes;
    if (e === null || t.position === void 0 || t.normal === void 0 || t.uv === void 0) {
      console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");
      return;
    }
    const i = t.position, s = t.normal, r = t.uv;
    this.hasAttribute("tangent") === !1 && this.setAttribute("tangent", new En(new Float32Array(4 * i.count), 4));
    const o = this.getAttribute("tangent"), a = [], l = [];
    for (let U = 0; U < i.count; U++)
      a[U] = new N(), l[U] = new N();
    const c = new N(), u = new N(), h = new N(), f = new Ve(), p = new Ve(), v = new Ve(), x = new N(), m = new N();
    function d(U, y, S) {
      c.fromBufferAttribute(i, U), u.fromBufferAttribute(i, y), h.fromBufferAttribute(i, S), f.fromBufferAttribute(r, U), p.fromBufferAttribute(r, y), v.fromBufferAttribute(r, S), u.sub(c), h.sub(c), p.sub(f), v.sub(f);
      const P = 1 / (p.x * v.y - v.x * p.y);
      isFinite(P) && (x.copy(u).multiplyScalar(v.y).addScaledVector(h, -p.y).multiplyScalar(P), m.copy(h).multiplyScalar(p.x).addScaledVector(u, -v.x).multiplyScalar(P), a[U].add(x), a[y].add(x), a[S].add(x), l[U].add(m), l[y].add(m), l[S].add(m));
    }
    let b = this.groups;
    b.length === 0 && (b = [{
      start: 0,
      count: e.count
    }]);
    for (let U = 0, y = b.length; U < y; ++U) {
      const S = b[U], P = S.start, L = S.count;
      for (let V = P, Z = P + L; V < Z; V += 3)
        d(
          e.getX(V + 0),
          e.getX(V + 1),
          e.getX(V + 2)
        );
    }
    const A = new N(), M = new N(), R = new N(), w = new N();
    function D(U) {
      R.fromBufferAttribute(s, U), w.copy(R);
      const y = a[U];
      A.copy(y), A.sub(R.multiplyScalar(R.dot(y))).normalize(), M.crossVectors(w, y);
      const P = M.dot(l[U]) < 0 ? -1 : 1;
      o.setXYZW(U, A.x, A.y, A.z, P);
    }
    for (let U = 0, y = b.length; U < y; ++U) {
      const S = b[U], P = S.start, L = S.count;
      for (let V = P, Z = P + L; V < Z; V += 3)
        D(e.getX(V + 0)), D(e.getX(V + 1)), D(e.getX(V + 2));
    }
  }
  /**
   * Computes vertex normals for the given vertex data. For indexed geometries, the method sets
   * each vertex normal to be the average of the face normals of the faces that share that vertex.
   * For non-indexed geometries, vertices are not shared, and the method sets each vertex normal
   * to be the same as the face normal.
   */
  computeVertexNormals() {
    const e = this.index, t = this.getAttribute("position");
    if (t !== void 0) {
      let i = this.getAttribute("normal");
      if (i === void 0)
        i = new En(new Float32Array(t.count * 3), 3), this.setAttribute("normal", i);
      else
        for (let f = 0, p = i.count; f < p; f++)
          i.setXYZ(f, 0, 0, 0);
      const s = new N(), r = new N(), o = new N(), a = new N(), l = new N(), c = new N(), u = new N(), h = new N();
      if (e)
        for (let f = 0, p = e.count; f < p; f += 3) {
          const v = e.getX(f + 0), x = e.getX(f + 1), m = e.getX(f + 2);
          s.fromBufferAttribute(t, v), r.fromBufferAttribute(t, x), o.fromBufferAttribute(t, m), u.subVectors(o, r), h.subVectors(s, r), u.cross(h), a.fromBufferAttribute(i, v), l.fromBufferAttribute(i, x), c.fromBufferAttribute(i, m), a.add(u), l.add(u), c.add(u), i.setXYZ(v, a.x, a.y, a.z), i.setXYZ(x, l.x, l.y, l.z), i.setXYZ(m, c.x, c.y, c.z);
        }
      else
        for (let f = 0, p = t.count; f < p; f += 3)
          s.fromBufferAttribute(t, f + 0), r.fromBufferAttribute(t, f + 1), o.fromBufferAttribute(t, f + 2), u.subVectors(o, r), h.subVectors(s, r), u.cross(h), i.setXYZ(f + 0, u.x, u.y, u.z), i.setXYZ(f + 1, u.x, u.y, u.z), i.setXYZ(f + 2, u.x, u.y, u.z);
      this.normalizeNormals(), i.needsUpdate = !0;
    }
  }
  /**
   * Ensures every normal vector in a geometry will have a magnitude of `1`. This will
   * correct lighting on the geometry surfaces.
   */
  normalizeNormals() {
    const e = this.attributes.normal;
    for (let t = 0, i = e.count; t < i; t++)
      wt.fromBufferAttribute(e, t), wt.normalize(), e.setXYZ(t, wt.x, wt.y, wt.z);
  }
  /**
   * Return a new non-index version of this indexed geometry. If the geometry
   * is already non-indexed, the method is a NOOP.
   *
   * @return {BufferGeometry} The non-indexed version of this indexed geometry.
   */
  toNonIndexed() {
    function e(a, l) {
      const c = a.array, u = a.itemSize, h = a.normalized, f = new c.constructor(l.length * u);
      let p = 0, v = 0;
      for (let x = 0, m = l.length; x < m; x++) {
        a.isInterleavedBufferAttribute ? p = l[x] * a.data.stride + a.offset : p = l[x] * u;
        for (let d = 0; d < u; d++)
          f[v++] = c[p++];
      }
      return new En(f, u, h);
    }
    if (this.index === null)
      return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."), this;
    const t = new Nt(), i = this.index.array, s = this.attributes;
    for (const a in s) {
      const l = s[a], c = e(l, i);
      t.setAttribute(a, c);
    }
    const r = this.morphAttributes;
    for (const a in r) {
      const l = [], c = r[a];
      for (let u = 0, h = c.length; u < h; u++) {
        const f = c[u], p = e(f, i);
        l.push(p);
      }
      t.morphAttributes[a] = l;
    }
    t.morphTargetsRelative = this.morphTargetsRelative;
    const o = this.groups;
    for (let a = 0, l = o.length; a < l; a++) {
      const c = o[a];
      t.addGroup(c.start, c.count, c.materialIndex);
    }
    return t;
  }
  /**
   * Serializes the geometry into JSON.
   *
   * @return {Object} A JSON object representing the serialized geometry.
   */
  toJSON() {
    const e = {
      metadata: {
        version: 4.7,
        type: "BufferGeometry",
        generator: "BufferGeometry.toJSON"
      }
    };
    if (e.uuid = this.uuid, e.type = this.type, this.name !== "" && (e.name = this.name), Object.keys(this.userData).length > 0 && (e.userData = this.userData), this.parameters !== void 0) {
      const l = this.parameters;
      for (const c in l)
        l[c] !== void 0 && (e[c] = l[c]);
      return e;
    }
    e.data = { attributes: {} };
    const t = this.index;
    t !== null && (e.data.index = {
      type: t.array.constructor.name,
      array: Array.prototype.slice.call(t.array)
    });
    const i = this.attributes;
    for (const l in i) {
      const c = i[l];
      e.data.attributes[l] = c.toJSON(e.data);
    }
    const s = {};
    let r = !1;
    for (const l in this.morphAttributes) {
      const c = this.morphAttributes[l], u = [];
      for (let h = 0, f = c.length; h < f; h++) {
        const p = c[h];
        u.push(p.toJSON(e.data));
      }
      u.length > 0 && (s[l] = u, r = !0);
    }
    r && (e.data.morphAttributes = s, e.data.morphTargetsRelative = this.morphTargetsRelative);
    const o = this.groups;
    o.length > 0 && (e.data.groups = JSON.parse(JSON.stringify(o)));
    const a = this.boundingSphere;
    return a !== null && (e.data.boundingSphere = a.toJSON()), e;
  }
  /**
   * Returns a new geometry with copied values from this instance.
   *
   * @return {BufferGeometry} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Copies the values of the given geometry to this instance.
   *
   * @param {BufferGeometry} source - The geometry to copy.
   * @return {BufferGeometry} A reference to this instance.
   */
  copy(e) {
    this.index = null, this.attributes = {}, this.morphAttributes = {}, this.groups = [], this.boundingBox = null, this.boundingSphere = null;
    const t = {};
    this.name = e.name;
    const i = e.index;
    i !== null && this.setIndex(i.clone());
    const s = e.attributes;
    for (const c in s) {
      const u = s[c];
      this.setAttribute(c, u.clone(t));
    }
    const r = e.morphAttributes;
    for (const c in r) {
      const u = [], h = r[c];
      for (let f = 0, p = h.length; f < p; f++)
        u.push(h[f].clone(t));
      this.morphAttributes[c] = u;
    }
    this.morphTargetsRelative = e.morphTargetsRelative;
    const o = e.groups;
    for (let c = 0, u = o.length; c < u; c++) {
      const h = o[c];
      this.addGroup(h.start, h.count, h.materialIndex);
    }
    const a = e.boundingBox;
    a !== null && (this.boundingBox = a.clone());
    const l = e.boundingSphere;
    return l !== null && (this.boundingSphere = l.clone()), this.drawRange.start = e.drawRange.start, this.drawRange.count = e.drawRange.count, this.userData = e.userData, this;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   *
   * @fires BufferGeometry#dispose
   */
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
}
const Gu = /* @__PURE__ */ new pt(), Di = /* @__PURE__ */ new na(), Zr = /* @__PURE__ */ new Nr(), Wu = /* @__PURE__ */ new N(), Jr = /* @__PURE__ */ new N(), Qr = /* @__PURE__ */ new N(), eo = /* @__PURE__ */ new N(), Ha = /* @__PURE__ */ new N(), to = /* @__PURE__ */ new N(), Xu = /* @__PURE__ */ new N(), no = /* @__PURE__ */ new N();
class vt extends Tt {
  /**
   * Constructs a new mesh.
   *
   * @param {BufferGeometry} [geometry] - The mesh geometry.
   * @param {Material|Array<Material>} [material] - The mesh material.
   */
  constructor(e = new Nt(), t = new Rn()) {
    super(), this.isMesh = !0, this.type = "Mesh", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.count = 1, this.updateMorphTargets();
  }
  copy(e, t) {
    return super.copy(e, t), e.morphTargetInfluences !== void 0 && (this.morphTargetInfluences = e.morphTargetInfluences.slice()), e.morphTargetDictionary !== void 0 && (this.morphTargetDictionary = Object.assign({}, e.morphTargetDictionary)), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
  }
  /**
   * Sets the values of {@link Mesh#morphTargetDictionary} and {@link Mesh#morphTargetInfluences}
   * to make sure existing morph targets can influence this 3D object.
   */
  updateMorphTargets() {
    const t = this.geometry.morphAttributes, i = Object.keys(t);
    if (i.length > 0) {
      const s = t[i[0]];
      if (s !== void 0) {
        this.morphTargetInfluences = [], this.morphTargetDictionary = {};
        for (let r = 0, o = s.length; r < o; r++) {
          const a = s[r].name || String(r);
          this.morphTargetInfluences.push(0), this.morphTargetDictionary[a] = r;
        }
      }
    }
  }
  /**
   * Returns the local-space position of the vertex at the given index, taking into
   * account the current animation state of both morph targets and skinning.
   *
   * @param {number} index - The vertex index.
   * @param {Vector3} target - The target object that is used to store the method's result.
   * @return {Vector3} The vertex position in local space.
   */
  getVertexPosition(e, t) {
    const i = this.geometry, s = i.attributes.position, r = i.morphAttributes.position, o = i.morphTargetsRelative;
    t.fromBufferAttribute(s, e);
    const a = this.morphTargetInfluences;
    if (r && a) {
      to.set(0, 0, 0);
      for (let l = 0, c = r.length; l < c; l++) {
        const u = a[l], h = r[l];
        u !== 0 && (Ha.fromBufferAttribute(h, e), o ? to.addScaledVector(Ha, u) : to.addScaledVector(Ha.sub(t), u));
      }
      t.add(to);
    }
    return t;
  }
  /**
   * Computes intersection points between a casted ray and this line.
   *
   * @param {Raycaster} raycaster - The raycaster.
   * @param {Array<Object>} intersects - The target array that holds the intersection points.
   */
  raycast(e, t) {
    const i = this.geometry, s = this.material, r = this.matrixWorld;
    s !== void 0 && (i.boundingSphere === null && i.computeBoundingSphere(), Zr.copy(i.boundingSphere), Zr.applyMatrix4(r), Di.copy(e.ray).recast(e.near), !(Zr.containsPoint(Di.origin) === !1 && (Di.intersectSphere(Zr, Wu) === null || Di.origin.distanceToSquared(Wu) > (e.far - e.near) ** 2)) && (Gu.copy(r).invert(), Di.copy(e.ray).applyMatrix4(Gu), !(i.boundingBox !== null && Di.intersectsBox(i.boundingBox) === !1) && this._computeIntersections(e, t, Di)));
  }
  _computeIntersections(e, t, i) {
    let s;
    const r = this.geometry, o = this.material, a = r.index, l = r.attributes.position, c = r.attributes.uv, u = r.attributes.uv1, h = r.attributes.normal, f = r.groups, p = r.drawRange;
    if (a !== null)
      if (Array.isArray(o))
        for (let v = 0, x = f.length; v < x; v++) {
          const m = f[v], d = o[m.materialIndex], b = Math.max(m.start, p.start), A = Math.min(a.count, Math.min(m.start + m.count, p.start + p.count));
          for (let M = b, R = A; M < R; M += 3) {
            const w = a.getX(M), D = a.getX(M + 1), U = a.getX(M + 2);
            s = io(this, d, e, i, c, u, h, w, D, U), s && (s.faceIndex = Math.floor(M / 3), s.face.materialIndex = m.materialIndex, t.push(s));
          }
        }
      else {
        const v = Math.max(0, p.start), x = Math.min(a.count, p.start + p.count);
        for (let m = v, d = x; m < d; m += 3) {
          const b = a.getX(m), A = a.getX(m + 1), M = a.getX(m + 2);
          s = io(this, o, e, i, c, u, h, b, A, M), s && (s.faceIndex = Math.floor(m / 3), t.push(s));
        }
      }
    else if (l !== void 0)
      if (Array.isArray(o))
        for (let v = 0, x = f.length; v < x; v++) {
          const m = f[v], d = o[m.materialIndex], b = Math.max(m.start, p.start), A = Math.min(l.count, Math.min(m.start + m.count, p.start + p.count));
          for (let M = b, R = A; M < R; M += 3) {
            const w = M, D = M + 1, U = M + 2;
            s = io(this, d, e, i, c, u, h, w, D, U), s && (s.faceIndex = Math.floor(M / 3), s.face.materialIndex = m.materialIndex, t.push(s));
          }
        }
      else {
        const v = Math.max(0, p.start), x = Math.min(l.count, p.start + p.count);
        for (let m = v, d = x; m < d; m += 3) {
          const b = m, A = m + 1, M = m + 2;
          s = io(this, o, e, i, c, u, h, b, A, M), s && (s.faceIndex = Math.floor(m / 3), t.push(s));
        }
      }
  }
}
function qg(n, e, t, i, s, r, o, a) {
  let l;
  if (e.side === Wt ? l = i.intersectTriangle(o, r, s, !0, a) : l = i.intersectTriangle(s, r, o, e.side === Si, a), l === null) return null;
  no.copy(a), no.applyMatrix4(n.matrixWorld);
  const c = t.ray.origin.distanceTo(no);
  return c < t.near || c > t.far ? null : {
    distance: c,
    point: no.clone(),
    object: n
  };
}
function io(n, e, t, i, s, r, o, a, l, c) {
  n.getVertexPosition(a, Jr), n.getVertexPosition(l, Qr), n.getVertexPosition(c, eo);
  const u = qg(n, e, t, i, Jr, Qr, eo, Xu);
  if (u) {
    const h = new N();
    fn.getBarycoord(Xu, Jr, Qr, eo, h), s && (u.uv = fn.getInterpolatedAttribute(s, a, l, c, h, new Ve())), r && (u.uv1 = fn.getInterpolatedAttribute(r, a, l, c, h, new Ve())), o && (u.normal = fn.getInterpolatedAttribute(o, a, l, c, h, new N()), u.normal.dot(i.direction) > 0 && u.normal.multiplyScalar(-1));
    const f = {
      a,
      b: l,
      c,
      normal: new N(),
      materialIndex: 0
    };
    fn.getNormal(Jr, Qr, eo, f.normal), u.face = f, u.barycoord = h;
  }
  return u;
}
class ji extends Nt {
  /**
   * Constructs a new box geometry.
   *
   * @param {number} [width=1] - The width. That is, the length of the edges parallel to the X axis.
   * @param {number} [height=1] - The height. That is, the length of the edges parallel to the Y axis.
   * @param {number} [depth=1] - The depth. That is, the length of the edges parallel to the Z axis.
   * @param {number} [widthSegments=1] - Number of segmented rectangular faces along the width of the sides.
   * @param {number} [heightSegments=1] - Number of segmented rectangular faces along the height of the sides.
   * @param {number} [depthSegments=1] - Number of segmented rectangular faces along the depth of the sides.
   */
  constructor(e = 1, t = 1, i = 1, s = 1, r = 1, o = 1) {
    super(), this.type = "BoxGeometry", this.parameters = {
      width: e,
      height: t,
      depth: i,
      widthSegments: s,
      heightSegments: r,
      depthSegments: o
    };
    const a = this;
    s = Math.floor(s), r = Math.floor(r), o = Math.floor(o);
    const l = [], c = [], u = [], h = [];
    let f = 0, p = 0;
    v("z", "y", "x", -1, -1, i, t, e, o, r, 0), v("z", "y", "x", 1, -1, i, t, -e, o, r, 1), v("x", "z", "y", 1, 1, e, i, t, s, o, 2), v("x", "z", "y", 1, -1, e, i, -t, s, o, 3), v("x", "y", "z", 1, -1, e, t, i, s, r, 4), v("x", "y", "z", -1, -1, e, t, -i, s, r, 5), this.setIndex(l), this.setAttribute("position", new mt(c, 3)), this.setAttribute("normal", new mt(u, 3)), this.setAttribute("uv", new mt(h, 2));
    function v(x, m, d, b, A, M, R, w, D, U, y) {
      const S = M / D, P = R / U, L = M / 2, V = R / 2, Z = w / 2, te = D + 1, $ = U + 1;
      let ie = 0, H = 0;
      const fe = new N();
      for (let xe = 0; xe < $; xe++) {
        const me = xe * P - V;
        for (let de = 0; de < te; de++) {
          const Le = de * S - L;
          fe[x] = Le * b, fe[m] = me * A, fe[d] = Z, c.push(fe.x, fe.y, fe.z), fe[x] = 0, fe[m] = 0, fe[d] = w > 0 ? 1 : -1, u.push(fe.x, fe.y, fe.z), h.push(de / D), h.push(1 - xe / U), ie += 1;
        }
      }
      for (let xe = 0; xe < U; xe++)
        for (let me = 0; me < D; me++) {
          const de = f + me + te * xe, Le = f + me + te * (xe + 1), tt = f + (me + 1) + te * (xe + 1), Ze = f + (me + 1) + te * xe;
          l.push(de, Le, Ze), l.push(Le, tt, Ze), H += 6;
        }
      a.addGroup(p, H, y), p += H, f += ie;
    }
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {BoxGeometry} A new instance.
   */
  static fromJSON(e) {
    return new ji(e.width, e.height, e.depth, e.widthSegments, e.heightSegments, e.depthSegments);
  }
}
function Bs(n) {
  const e = {};
  for (const t in n) {
    e[t] = {};
    for (const i in n[t]) {
      const s = n[t][i];
      s && (s.isColor || s.isMatrix3 || s.isMatrix4 || s.isVector2 || s.isVector3 || s.isVector4 || s.isTexture || s.isQuaternion) ? s.isRenderTargetTexture ? (console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."), e[t][i] = null) : e[t][i] = s.clone() : Array.isArray(s) ? e[t][i] = s.slice() : e[t][i] = s;
    }
  }
  return e;
}
function zt(n) {
  const e = {};
  for (let t = 0; t < n.length; t++) {
    const i = Bs(n[t]);
    for (const s in i)
      e[s] = i[s];
  }
  return e;
}
function jg(n) {
  const e = [];
  for (let t = 0; t < n.length; t++)
    e.push(n[t].clone());
  return e;
}
function vd(n) {
  const e = n.getRenderTarget();
  return e === null ? n.outputColorSpace : e.isXRRenderTarget === !0 ? e.texture.colorSpace : Qe.workingColorSpace;
}
const Kg = { clone: Bs, merge: zt };
var $g = `void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`, Zg = `void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;
class yi extends Ji {
  /**
   * Constructs a new shader material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isShaderMaterial = !0, this.type = "ShaderMaterial", this.defines = {}, this.uniforms = {}, this.uniformsGroups = [], this.vertexShader = $g, this.fragmentShader = Zg, this.linewidth = 1, this.wireframe = !1, this.wireframeLinewidth = 1, this.fog = !1, this.lights = !1, this.clipping = !1, this.forceSinglePass = !0, this.extensions = {
      clipCullDistance: !1,
      // set to use vertex shader clipping
      multiDraw: !1
      // set to use vertex shader multi_draw / enable gl_DrawID
    }, this.defaultAttributeValues = {
      color: [1, 1, 1],
      uv: [0, 0],
      uv1: [0, 0]
    }, this.index0AttributeName = void 0, this.uniformsNeedUpdate = !1, this.glslVersion = null, e !== void 0 && this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.fragmentShader = e.fragmentShader, this.vertexShader = e.vertexShader, this.uniforms = Bs(e.uniforms), this.uniformsGroups = jg(e.uniformsGroups), this.defines = Object.assign({}, e.defines), this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this.fog = e.fog, this.lights = e.lights, this.clipping = e.clipping, this.extensions = Object.assign({}, e.extensions), this.glslVersion = e.glslVersion, this;
  }
  toJSON(e) {
    const t = super.toJSON(e);
    t.glslVersion = this.glslVersion, t.uniforms = {};
    for (const s in this.uniforms) {
      const o = this.uniforms[s].value;
      o && o.isTexture ? t.uniforms[s] = {
        type: "t",
        value: o.toJSON(e).uuid
      } : o && o.isColor ? t.uniforms[s] = {
        type: "c",
        value: o.getHex()
      } : o && o.isVector2 ? t.uniforms[s] = {
        type: "v2",
        value: o.toArray()
      } : o && o.isVector3 ? t.uniforms[s] = {
        type: "v3",
        value: o.toArray()
      } : o && o.isVector4 ? t.uniforms[s] = {
        type: "v4",
        value: o.toArray()
      } : o && o.isMatrix3 ? t.uniforms[s] = {
        type: "m3",
        value: o.toArray()
      } : o && o.isMatrix4 ? t.uniforms[s] = {
        type: "m4",
        value: o.toArray()
      } : t.uniforms[s] = {
        value: o
      };
    }
    Object.keys(this.defines).length > 0 && (t.defines = this.defines), t.vertexShader = this.vertexShader, t.fragmentShader = this.fragmentShader, t.lights = this.lights, t.clipping = this.clipping;
    const i = {};
    for (const s in this.extensions)
      this.extensions[s] === !0 && (i[s] = !0);
    return Object.keys(i).length > 0 && (t.extensions = i), t;
  }
}
class xd extends Tt {
  /**
   * Constructs a new camera.
   */
  constructor() {
    super(), this.isCamera = !0, this.type = "Camera", this.matrixWorldInverse = new pt(), this.projectionMatrix = new pt(), this.projectionMatrixInverse = new pt(), this.coordinateSystem = Nn, this._reversedDepth = !1;
  }
  /**
   * The flag that indicates whether the camera uses a reversed depth buffer.
   *
   * @type {boolean}
   * @default false
   */
  get reversedDepth() {
    return this._reversedDepth;
  }
  copy(e, t) {
    return super.copy(e, t), this.matrixWorldInverse.copy(e.matrixWorldInverse), this.projectionMatrix.copy(e.projectionMatrix), this.projectionMatrixInverse.copy(e.projectionMatrixInverse), this.coordinateSystem = e.coordinateSystem, this;
  }
  /**
   * Returns a vector representing the ("look") direction of the 3D object in world space.
   *
   * This method is overwritten since cameras have a different forward vector compared to other
   * 3D objects. A camera looks down its local, negative z-axis by default.
   *
   * @param {Vector3} target - The target vector the result is stored to.
   * @return {Vector3} The 3D object's direction in world space.
   */
  getWorldDirection(e) {
    return super.getWorldDirection(e).negate();
  }
  updateMatrixWorld(e) {
    super.updateMatrixWorld(e), this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }
  updateWorldMatrix(e, t) {
    super.updateWorldMatrix(e, t), this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
const fi = /* @__PURE__ */ new N(), Yu = /* @__PURE__ */ new Ve(), qu = /* @__PURE__ */ new Ve();
class nn extends xd {
  /**
   * Constructs a new perspective camera.
   *
   * @param {number} [fov=50] - The vertical field of view.
   * @param {number} [aspect=1] - The aspect ratio.
   * @param {number} [near=0.1] - The camera's near plane.
   * @param {number} [far=2000] - The camera's far plane.
   */
  constructor(e = 50, t = 1, i = 0.1, s = 2e3) {
    super(), this.isPerspectiveCamera = !0, this.type = "PerspectiveCamera", this.fov = e, this.zoom = 1, this.near = i, this.far = s, this.focus = 10, this.aspect = t, this.view = null, this.filmGauge = 35, this.filmOffset = 0, this.updateProjectionMatrix();
  }
  copy(e, t) {
    return super.copy(e, t), this.fov = e.fov, this.zoom = e.zoom, this.near = e.near, this.far = e.far, this.focus = e.focus, this.aspect = e.aspect, this.view = e.view === null ? null : Object.assign({}, e.view), this.filmGauge = e.filmGauge, this.filmOffset = e.filmOffset, this;
  }
  /**
   * Sets the FOV by focal length in respect to the current {@link PerspectiveCamera#filmGauge}.
   *
   * The default film gauge is 35, so that the focal length can be specified for
   * a 35mm (full frame) camera.
   *
   * @param {number} focalLength - Values for focal length and film gauge must have the same unit.
   */
  setFocalLength(e) {
    const t = 0.5 * this.getFilmHeight() / e;
    this.fov = Ql * 2 * Math.atan(t), this.updateProjectionMatrix();
  }
  /**
   * Returns the focal length from the current {@link PerspectiveCamera#fov} and
   * {@link PerspectiveCamera#filmGauge}.
   *
   * @return {number} The computed focal length.
   */
  getFocalLength() {
    const e = Math.tan(dr * 0.5 * this.fov);
    return 0.5 * this.getFilmHeight() / e;
  }
  /**
   * Returns the current vertical field of view angle in degrees considering {@link PerspectiveCamera#zoom}.
   *
   * @return {number} The effective FOV.
   */
  getEffectiveFOV() {
    return Ql * 2 * Math.atan(
      Math.tan(dr * 0.5 * this.fov) / this.zoom
    );
  }
  /**
   * Returns the width of the image on the film. If {@link PerspectiveCamera#aspect} is greater than or
   * equal to one (landscape format), the result equals {@link PerspectiveCamera#filmGauge}.
   *
   * @return {number} The film width.
   */
  getFilmWidth() {
    return this.filmGauge * Math.min(this.aspect, 1);
  }
  /**
   * Returns the height of the image on the film. If {@link PerspectiveCamera#aspect} is greater than or
   * equal to one (landscape format), the result equals {@link PerspectiveCamera#filmGauge}.
   *
   * @return {number} The film width.
   */
  getFilmHeight() {
    return this.filmGauge / Math.max(this.aspect, 1);
  }
  /**
   * Computes the 2D bounds of the camera's viewable rectangle at a given distance along the viewing direction.
   * Sets `minTarget` and `maxTarget` to the coordinates of the lower-left and upper-right corners of the view rectangle.
   *
   * @param {number} distance - The viewing distance.
   * @param {Vector2} minTarget - The lower-left corner of the view rectangle is written into this vector.
   * @param {Vector2} maxTarget - The upper-right corner of the view rectangle is written into this vector.
   */
  getViewBounds(e, t, i) {
    fi.set(-1, -1, 0.5).applyMatrix4(this.projectionMatrixInverse), t.set(fi.x, fi.y).multiplyScalar(-e / fi.z), fi.set(1, 1, 0.5).applyMatrix4(this.projectionMatrixInverse), i.set(fi.x, fi.y).multiplyScalar(-e / fi.z);
  }
  /**
   * Computes the width and height of the camera's viewable rectangle at a given distance along the viewing direction.
   *
   * @param {number} distance - The viewing distance.
   * @param {Vector2} target - The target vector that is used to store result where x is width and y is height.
   * @returns {Vector2} The view size.
   */
  getViewSize(e, t) {
    return this.getViewBounds(e, Yu, qu), t.subVectors(qu, Yu);
  }
  /**
   * Sets an offset in a larger frustum. This is useful for multi-window or
   * multi-monitor/multi-machine setups.
   *
   * For example, if you have 3x2 monitors and each monitor is 1920x1080 and
   * the monitors are in grid like this
   *```
   *   +---+---+---+
   *   | A | B | C |
   *   +---+---+---+
   *   | D | E | F |
   *   +---+---+---+
   *```
   * then for each monitor you would call it like this:
   *```js
   * const w = 1920;
   * const h = 1080;
   * const fullWidth = w * 3;
   * const fullHeight = h * 2;
   *
   * // --A--
   * camera.setViewOffset( fullWidth, fullHeight, w * 0, h * 0, w, h );
   * // --B--
   * camera.setViewOffset( fullWidth, fullHeight, w * 1, h * 0, w, h );
   * // --C--
   * camera.setViewOffset( fullWidth, fullHeight, w * 2, h * 0, w, h );
   * // --D--
   * camera.setViewOffset( fullWidth, fullHeight, w * 0, h * 1, w, h );
   * // --E--
   * camera.setViewOffset( fullWidth, fullHeight, w * 1, h * 1, w, h );
   * // --F--
   * camera.setViewOffset( fullWidth, fullHeight, w * 2, h * 1, w, h );
   * ```
   *
   * Note there is no reason monitors have to be the same size or in a grid.
   *
   * @param {number} fullWidth - The full width of multiview setup.
   * @param {number} fullHeight - The full height of multiview setup.
   * @param {number} x - The horizontal offset of the subcamera.
   * @param {number} y - The vertical offset of the subcamera.
   * @param {number} width - The width of subcamera.
   * @param {number} height - The height of subcamera.
   */
  setViewOffset(e, t, i, s, r, o) {
    this.aspect = e / t, this.view === null && (this.view = {
      enabled: !0,
      fullWidth: 1,
      fullHeight: 1,
      offsetX: 0,
      offsetY: 0,
      width: 1,
      height: 1
    }), this.view.enabled = !0, this.view.fullWidth = e, this.view.fullHeight = t, this.view.offsetX = i, this.view.offsetY = s, this.view.width = r, this.view.height = o, this.updateProjectionMatrix();
  }
  /**
   * Removes the view offset from the projection matrix.
   */
  clearViewOffset() {
    this.view !== null && (this.view.enabled = !1), this.updateProjectionMatrix();
  }
  /**
   * Updates the camera's projection matrix. Must be called after any change of
   * camera properties.
   */
  updateProjectionMatrix() {
    const e = this.near;
    let t = e * Math.tan(dr * 0.5 * this.fov) / this.zoom, i = 2 * t, s = this.aspect * i, r = -0.5 * s;
    const o = this.view;
    if (this.view !== null && this.view.enabled) {
      const l = o.fullWidth, c = o.fullHeight;
      r += o.offsetX * s / l, t -= o.offsetY * i / c, s *= o.width / l, i *= o.height / c;
    }
    const a = this.filmOffset;
    a !== 0 && (r += e * a / this.getFilmWidth()), this.projectionMatrix.makePerspective(r, r + s, t, t - i, e, this.far, this.coordinateSystem, this.reversedDepth), this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return t.object.fov = this.fov, t.object.zoom = this.zoom, t.object.near = this.near, t.object.far = this.far, t.object.focus = this.focus, t.object.aspect = this.aspect, this.view !== null && (t.object.view = Object.assign({}, this.view)), t.object.filmGauge = this.filmGauge, t.object.filmOffset = this.filmOffset, t;
  }
}
const ds = -90, ps = 1;
class Jg extends Tt {
  /**
   * Constructs a new cube camera.
   *
   * @param {number} near - The camera's near plane.
   * @param {number} far - The camera's far plane.
   * @param {WebGLCubeRenderTarget} renderTarget - The cube render target.
   */
  constructor(e, t, i) {
    super(), this.type = "CubeCamera", this.renderTarget = i, this.coordinateSystem = null, this.activeMipmapLevel = 0;
    const s = new nn(ds, ps, e, t);
    s.layers = this.layers, this.add(s);
    const r = new nn(ds, ps, e, t);
    r.layers = this.layers, this.add(r);
    const o = new nn(ds, ps, e, t);
    o.layers = this.layers, this.add(o);
    const a = new nn(ds, ps, e, t);
    a.layers = this.layers, this.add(a);
    const l = new nn(ds, ps, e, t);
    l.layers = this.layers, this.add(l);
    const c = new nn(ds, ps, e, t);
    c.layers = this.layers, this.add(c);
  }
  /**
   * Must be called when the coordinate system of the cube camera is changed.
   */
  updateCoordinateSystem() {
    const e = this.coordinateSystem, t = this.children.concat(), [i, s, r, o, a, l] = t;
    for (const c of t) this.remove(c);
    if (e === Nn)
      i.up.set(0, 1, 0), i.lookAt(1, 0, 0), s.up.set(0, 1, 0), s.lookAt(-1, 0, 0), r.up.set(0, 0, -1), r.lookAt(0, 1, 0), o.up.set(0, 0, 1), o.lookAt(0, -1, 0), a.up.set(0, 1, 0), a.lookAt(0, 0, 1), l.up.set(0, 1, 0), l.lookAt(0, 0, -1);
    else if (e === zo)
      i.up.set(0, -1, 0), i.lookAt(-1, 0, 0), s.up.set(0, -1, 0), s.lookAt(1, 0, 0), r.up.set(0, 0, 1), r.lookAt(0, 1, 0), o.up.set(0, 0, -1), o.lookAt(0, -1, 0), a.up.set(0, -1, 0), a.lookAt(0, 0, 1), l.up.set(0, -1, 0), l.lookAt(0, 0, -1);
    else
      throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: " + e);
    for (const c of t)
      this.add(c), c.updateMatrixWorld();
  }
  /**
   * Calling this method will render the given scene with the given renderer
   * into the cube render target of the camera.
   *
   * @param {(Renderer|WebGLRenderer)} renderer - The renderer.
   * @param {Scene} scene - The scene to render.
   */
  update(e, t) {
    this.parent === null && this.updateMatrixWorld();
    const { renderTarget: i, activeMipmapLevel: s } = this;
    this.coordinateSystem !== e.coordinateSystem && (this.coordinateSystem = e.coordinateSystem, this.updateCoordinateSystem());
    const [r, o, a, l, c, u] = this.children, h = e.getRenderTarget(), f = e.getActiveCubeFace(), p = e.getActiveMipmapLevel(), v = e.xr.enabled;
    e.xr.enabled = !1;
    const x = i.texture.generateMipmaps;
    i.texture.generateMipmaps = !1, e.setRenderTarget(i, 0, s), e.render(t, r), e.setRenderTarget(i, 1, s), e.render(t, o), e.setRenderTarget(i, 2, s), e.render(t, a), e.setRenderTarget(i, 3, s), e.render(t, l), e.setRenderTarget(i, 4, s), e.render(t, c), i.texture.generateMipmaps = x, e.setRenderTarget(i, 5, s), e.render(t, u), e.setRenderTarget(h, f, p), e.xr.enabled = v, i.texture.needsPMREMUpdate = !0;
  }
}
class Md extends $t {
  /**
   * Constructs a new cube texture.
   *
   * @param {Array<Image>} [images=[]] - An array holding a image for each side of a cube.
   * @param {number} [mapping=CubeReflectionMapping] - The texture mapping.
   * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
   * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
   * @param {number} [magFilter=LinearFilter] - The mag filter value.
   * @param {number} [minFilter=LinearMipmapLinearFilter] - The min filter value.
   * @param {number} [format=RGBAFormat] - The texture format.
   * @param {number} [type=UnsignedByteType] - The texture type.
   * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
   * @param {string} [colorSpace=NoColorSpace] - The color space value.
   */
  constructor(e = [], t = Ns, i, s, r, o, a, l, c, u) {
    super(e, t, i, s, r, o, a, l, c, u), this.isCubeTexture = !0, this.flipY = !1;
  }
  /**
   * Alias for {@link CubeTexture#image}.
   *
   * @type {Array<Image>}
   */
  get images() {
    return this.image;
  }
  set images(e) {
    this.image = e;
  }
}
class Qg extends qi {
  /**
   * Constructs a new cube render target.
   *
   * @param {number} [size=1] - The size of the render target.
   * @param {RenderTarget~Options} [options] - The configuration object.
   */
  constructor(e = 1, t = {}) {
    super(e, e, t), this.isWebGLCubeRenderTarget = !0;
    const i = { width: e, height: e, depth: 1 }, s = [i, i, i, i, i, i];
    this.texture = new Md(s), this._setTextureOptions(t), this.texture.isRenderTargetTexture = !0;
  }
  /**
   * Converts the given equirectangular texture to a cube map.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   * @param {Texture} texture - The equirectangular texture.
   * @return {WebGLCubeRenderTarget} A reference to this cube render target.
   */
  fromEquirectangularTexture(e, t) {
    this.texture.type = t.type, this.texture.colorSpace = t.colorSpace, this.texture.generateMipmaps = t.generateMipmaps, this.texture.minFilter = t.minFilter, this.texture.magFilter = t.magFilter;
    const i = {
      uniforms: {
        tEquirect: { value: null }
      },
      vertexShader: (
        /* glsl */
        `

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`
      ),
      fragmentShader: (
        /* glsl */
        `

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`
      )
    }, s = new ji(5, 5, 5), r = new yi({
      name: "CubemapFromEquirect",
      uniforms: Bs(i.uniforms),
      vertexShader: i.vertexShader,
      fragmentShader: i.fragmentShader,
      side: Wt,
      blending: vi
    });
    r.uniforms.tEquirect.value = t;
    const o = new vt(s, r), a = t.minFilter;
    return t.minFilter === Vi && (t.minFilter = Un), new Jg(1, 10, this).update(e, o), t.minFilter = a, o.geometry.dispose(), o.material.dispose(), this;
  }
  /**
   * Clears this cube render target.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   * @param {boolean} [color=true] - Whether the color buffer should be cleared or not.
   * @param {boolean} [depth=true] - Whether the depth buffer should be cleared or not.
   * @param {boolean} [stencil=true] - Whether the stencil buffer should be cleared or not.
   */
  clear(e, t = !0, i = !0, s = !0) {
    const r = e.getRenderTarget();
    for (let o = 0; o < 6; o++)
      e.setRenderTarget(this, o), e.clear(t, i, s);
    e.setRenderTarget(r);
  }
}
class Dn extends Tt {
  constructor() {
    super(), this.isGroup = !0, this.type = "Group";
  }
}
const e0 = { type: "move" };
class Va {
  /**
   * Constructs a new XR controller.
   */
  constructor() {
    this._targetRay = null, this._grip = null, this._hand = null;
  }
  /**
   * Returns a group representing the hand space of the XR controller.
   *
   * @return {Group} A group representing the hand space of the XR controller.
   */
  getHandSpace() {
    return this._hand === null && (this._hand = new Dn(), this._hand.matrixAutoUpdate = !1, this._hand.visible = !1, this._hand.joints = {}, this._hand.inputState = { pinching: !1 }), this._hand;
  }
  /**
   * Returns a group representing the target ray space of the XR controller.
   *
   * @return {Group} A group representing the target ray space of the XR controller.
   */
  getTargetRaySpace() {
    return this._targetRay === null && (this._targetRay = new Dn(), this._targetRay.matrixAutoUpdate = !1, this._targetRay.visible = !1, this._targetRay.hasLinearVelocity = !1, this._targetRay.linearVelocity = new N(), this._targetRay.hasAngularVelocity = !1, this._targetRay.angularVelocity = new N()), this._targetRay;
  }
  /**
   * Returns a group representing the grip space of the XR controller.
   *
   * @return {Group} A group representing the grip space of the XR controller.
   */
  getGripSpace() {
    return this._grip === null && (this._grip = new Dn(), this._grip.matrixAutoUpdate = !1, this._grip.visible = !1, this._grip.hasLinearVelocity = !1, this._grip.linearVelocity = new N(), this._grip.hasAngularVelocity = !1, this._grip.angularVelocity = new N()), this._grip;
  }
  /**
   * Dispatches the given event to the groups representing
   * the different coordinate spaces of the XR controller.
   *
   * @param {Object} event - The event to dispatch.
   * @return {WebXRController} A reference to this instance.
   */
  dispatchEvent(e) {
    return this._targetRay !== null && this._targetRay.dispatchEvent(e), this._grip !== null && this._grip.dispatchEvent(e), this._hand !== null && this._hand.dispatchEvent(e), this;
  }
  /**
   * Connects the controller with the given XR input source.
   *
   * @param {XRInputSource} inputSource - The input source.
   * @return {WebXRController} A reference to this instance.
   */
  connect(e) {
    if (e && e.hand) {
      const t = this._hand;
      if (t)
        for (const i of e.hand.values())
          this._getHandJoint(t, i);
    }
    return this.dispatchEvent({ type: "connected", data: e }), this;
  }
  /**
   * Disconnects the controller from the given XR input source.
   *
   * @param {XRInputSource} inputSource - The input source.
   * @return {WebXRController} A reference to this instance.
   */
  disconnect(e) {
    return this.dispatchEvent({ type: "disconnected", data: e }), this._targetRay !== null && (this._targetRay.visible = !1), this._grip !== null && (this._grip.visible = !1), this._hand !== null && (this._hand.visible = !1), this;
  }
  /**
   * Updates the controller with the given input source, XR frame and reference space.
   * This updates the transformations of the groups that represent the different
   * coordinate systems of the controller.
   *
   * @param {XRInputSource} inputSource - The input source.
   * @param {XRFrame} frame - The XR frame.
   * @param {XRReferenceSpace} referenceSpace - The reference space.
   * @return {WebXRController} A reference to this instance.
   */
  update(e, t, i) {
    let s = null, r = null, o = null;
    const a = this._targetRay, l = this._grip, c = this._hand;
    if (e && t.session.visibilityState !== "visible-blurred") {
      if (c && e.hand) {
        o = !0;
        for (const x of e.hand.values()) {
          const m = t.getJointPose(x, i), d = this._getHandJoint(c, x);
          m !== null && (d.matrix.fromArray(m.transform.matrix), d.matrix.decompose(d.position, d.rotation, d.scale), d.matrixWorldNeedsUpdate = !0, d.jointRadius = m.radius), d.visible = m !== null;
        }
        const u = c.joints["index-finger-tip"], h = c.joints["thumb-tip"], f = u.position.distanceTo(h.position), p = 0.02, v = 5e-3;
        c.inputState.pinching && f > p + v ? (c.inputState.pinching = !1, this.dispatchEvent({
          type: "pinchend",
          handedness: e.handedness,
          target: this
        })) : !c.inputState.pinching && f <= p - v && (c.inputState.pinching = !0, this.dispatchEvent({
          type: "pinchstart",
          handedness: e.handedness,
          target: this
        }));
      } else
        l !== null && e.gripSpace && (r = t.getPose(e.gripSpace, i), r !== null && (l.matrix.fromArray(r.transform.matrix), l.matrix.decompose(l.position, l.rotation, l.scale), l.matrixWorldNeedsUpdate = !0, r.linearVelocity ? (l.hasLinearVelocity = !0, l.linearVelocity.copy(r.linearVelocity)) : l.hasLinearVelocity = !1, r.angularVelocity ? (l.hasAngularVelocity = !0, l.angularVelocity.copy(r.angularVelocity)) : l.hasAngularVelocity = !1));
      a !== null && (s = t.getPose(e.targetRaySpace, i), s === null && r !== null && (s = r), s !== null && (a.matrix.fromArray(s.transform.matrix), a.matrix.decompose(a.position, a.rotation, a.scale), a.matrixWorldNeedsUpdate = !0, s.linearVelocity ? (a.hasLinearVelocity = !0, a.linearVelocity.copy(s.linearVelocity)) : a.hasLinearVelocity = !1, s.angularVelocity ? (a.hasAngularVelocity = !0, a.angularVelocity.copy(s.angularVelocity)) : a.hasAngularVelocity = !1, this.dispatchEvent(e0)));
    }
    return a !== null && (a.visible = s !== null), l !== null && (l.visible = r !== null), c !== null && (c.visible = o !== null), this;
  }
  /**
   * Returns a group representing the hand joint for the given input joint.
   *
   * @private
   * @param {Group} hand - The group representing the hand space.
   * @param {XRJointSpace} inputjoint - The hand joint data.
   * @return {Group} A group representing the hand joint for the given input joint.
   */
  _getHandJoint(e, t) {
    if (e.joints[t.jointName] === void 0) {
      const i = new Dn();
      i.matrixAutoUpdate = !1, i.visible = !1, e.joints[t.jointName] = i, e.add(i);
    }
    return e.joints[t.jointName];
  }
}
class t0 extends Tt {
  /**
   * Constructs a new scene.
   */
  constructor() {
    super(), this.isScene = !0, this.type = "Scene", this.background = null, this.environment = null, this.fog = null, this.backgroundBlurriness = 0, this.backgroundIntensity = 1, this.backgroundRotation = new zn(), this.environmentIntensity = 1, this.environmentRotation = new zn(), this.overrideMaterial = null, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
  }
  copy(e, t) {
    return super.copy(e, t), e.background !== null && (this.background = e.background.clone()), e.environment !== null && (this.environment = e.environment.clone()), e.fog !== null && (this.fog = e.fog.clone()), this.backgroundBlurriness = e.backgroundBlurriness, this.backgroundIntensity = e.backgroundIntensity, this.backgroundRotation.copy(e.backgroundRotation), this.environmentIntensity = e.environmentIntensity, this.environmentRotation.copy(e.environmentRotation), e.overrideMaterial !== null && (this.overrideMaterial = e.overrideMaterial.clone()), this.matrixAutoUpdate = e.matrixAutoUpdate, this;
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return this.fog !== null && (t.object.fog = this.fog.toJSON()), this.backgroundBlurriness > 0 && (t.object.backgroundBlurriness = this.backgroundBlurriness), this.backgroundIntensity !== 1 && (t.object.backgroundIntensity = this.backgroundIntensity), t.object.backgroundRotation = this.backgroundRotation.toArray(), this.environmentIntensity !== 1 && (t.object.environmentIntensity = this.environmentIntensity), t.object.environmentRotation = this.environmentRotation.toArray(), t;
  }
}
const ka = /* @__PURE__ */ new N(), n0 = /* @__PURE__ */ new N(), i0 = /* @__PURE__ */ new Ye();
class mi {
  /**
   * Constructs a new plane.
   *
   * @param {Vector3} [normal=(1,0,0)] - A unit length vector defining the normal of the plane.
   * @param {number} [constant=0] - The signed distance from the origin to the plane.
   */
  constructor(e = new N(1, 0, 0), t = 0) {
    this.isPlane = !0, this.normal = e, this.constant = t;
  }
  /**
   * Sets the plane components by copying the given values.
   *
   * @param {Vector3} normal - The normal.
   * @param {number} constant - The constant.
   * @return {Plane} A reference to this plane.
   */
  set(e, t) {
    return this.normal.copy(e), this.constant = t, this;
  }
  /**
   * Sets the plane components by defining `x`, `y`, `z` as the
   * plane normal and `w` as the constant.
   *
   * @param {number} x - The value for the normal's x component.
   * @param {number} y - The value for the normal's y component.
   * @param {number} z - The value for the normal's z component.
   * @param {number} w - The constant value.
   * @return {Plane} A reference to this plane.
   */
  setComponents(e, t, i, s) {
    return this.normal.set(e, t, i), this.constant = s, this;
  }
  /**
   * Sets the plane from the given normal and coplanar point (that is a point
   * that lies onto the plane).
   *
   * @param {Vector3} normal - The normal.
   * @param {Vector3} point - A coplanar point.
   * @return {Plane} A reference to this plane.
   */
  setFromNormalAndCoplanarPoint(e, t) {
    return this.normal.copy(e), this.constant = -t.dot(this.normal), this;
  }
  /**
   * Sets the plane from three coplanar points. The winding order is
   * assumed to be counter-clockwise, and determines the direction of
   * the plane normal.
   *
   * @param {Vector3} a - The first coplanar point.
   * @param {Vector3} b - The second coplanar point.
   * @param {Vector3} c - The third coplanar point.
   * @return {Plane} A reference to this plane.
   */
  setFromCoplanarPoints(e, t, i) {
    const s = ka.subVectors(i, t).cross(n0.subVectors(e, t)).normalize();
    return this.setFromNormalAndCoplanarPoint(s, e), this;
  }
  /**
   * Copies the values of the given plane to this instance.
   *
   * @param {Plane} plane - The plane to copy.
   * @return {Plane} A reference to this plane.
   */
  copy(e) {
    return this.normal.copy(e.normal), this.constant = e.constant, this;
  }
  /**
   * Normalizes the plane normal and adjusts the constant accordingly.
   *
   * @return {Plane} A reference to this plane.
   */
  normalize() {
    const e = 1 / this.normal.length();
    return this.normal.multiplyScalar(e), this.constant *= e, this;
  }
  /**
   * Negates both the plane normal and the constant.
   *
   * @return {Plane} A reference to this plane.
   */
  negate() {
    return this.constant *= -1, this.normal.negate(), this;
  }
  /**
   * Returns the signed distance from the given point to this plane.
   *
   * @param {Vector3} point - The point to compute the distance for.
   * @return {number} The signed distance.
   */
  distanceToPoint(e) {
    return this.normal.dot(e) + this.constant;
  }
  /**
   * Returns the signed distance from the given sphere to this plane.
   *
   * @param {Sphere} sphere - The sphere to compute the distance for.
   * @return {number} The signed distance.
   */
  distanceToSphere(e) {
    return this.distanceToPoint(e.center) - e.radius;
  }
  /**
   * Projects a the given point onto the plane.
   *
   * @param {Vector3} point - The point to project.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The projected point on the plane.
   */
  projectPoint(e, t) {
    return t.copy(e).addScaledVector(this.normal, -this.distanceToPoint(e));
  }
  /**
   * Returns the intersection point of the passed line and the plane. Returns
   * `null` if the line does not intersect. Returns the line's starting point if
   * the line is coplanar with the plane.
   *
   * @param {Line3} line - The line to compute the intersection for.
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {?Vector3} The intersection point.
   */
  intersectLine(e, t) {
    const i = e.delta(ka), s = this.normal.dot(i);
    if (s === 0)
      return this.distanceToPoint(e.start) === 0 ? t.copy(e.start) : null;
    const r = -(e.start.dot(this.normal) + this.constant) / s;
    return r < 0 || r > 1 ? null : t.copy(e.start).addScaledVector(i, r);
  }
  /**
   * Returns `true` if the given line segment intersects with (passes through) the plane.
   *
   * @param {Line3} line - The line to test.
   * @return {boolean} Whether the given line segment intersects with the plane or not.
   */
  intersectsLine(e) {
    const t = this.distanceToPoint(e.start), i = this.distanceToPoint(e.end);
    return t < 0 && i > 0 || i < 0 && t > 0;
  }
  /**
   * Returns `true` if the given bounding box intersects with the plane.
   *
   * @param {Box3} box - The bounding box to test.
   * @return {boolean} Whether the given bounding box intersects with the plane or not.
   */
  intersectsBox(e) {
    return e.intersectsPlane(this);
  }
  /**
   * Returns `true` if the given bounding sphere intersects with the plane.
   *
   * @param {Sphere} sphere - The bounding sphere to test.
   * @return {boolean} Whether the given bounding sphere intersects with the plane or not.
   */
  intersectsSphere(e) {
    return e.intersectsPlane(this);
  }
  /**
   * Returns a coplanar vector to the plane, by calculating the
   * projection of the normal at the origin onto the plane.
   *
   * @param {Vector3} target - The target vector that is used to store the method's result.
   * @return {Vector3} The coplanar point.
   */
  coplanarPoint(e) {
    return e.copy(this.normal).multiplyScalar(-this.constant);
  }
  /**
   * Apply a 4x4 matrix to the plane. The matrix must be an affine, homogeneous transform.
   *
   * The optional normal matrix can be pre-computed like so:
   * ```js
   * const optionalNormalMatrix = new THREE.Matrix3().getNormalMatrix( matrix );
   * ```
   *
   * @param {Matrix4} matrix - The transformation matrix.
   * @param {Matrix4} [optionalNormalMatrix] - A pre-computed normal matrix.
   * @return {Plane} A reference to this plane.
   */
  applyMatrix4(e, t) {
    const i = t || i0.getNormalMatrix(e), s = this.coplanarPoint(ka).applyMatrix4(e), r = this.normal.applyMatrix3(i).normalize();
    return this.constant = -s.dot(r), this;
  }
  /**
   * Translates the plane by the distance defined by the given offset vector.
   * Note that this only affects the plane constant and will not affect the normal vector.
   *
   * @param {Vector3} offset - The offset vector.
   * @return {Plane} A reference to this plane.
   */
  translate(e) {
    return this.constant -= e.dot(this.normal), this;
  }
  /**
   * Returns `true` if this plane is equal with the given one.
   *
   * @param {Plane} plane - The plane to test for equality.
   * @return {boolean} Whether this plane is equal with the given one.
   */
  equals(e) {
    return e.normal.equals(this.normal) && e.constant === this.constant;
  }
  /**
   * Returns a new plane with copied values from this instance.
   *
   * @return {Plane} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
}
const Li = /* @__PURE__ */ new Nr(), s0 = /* @__PURE__ */ new Ve(0.5, 0.5), so = /* @__PURE__ */ new N();
class Ac {
  /**
   * Constructs a new frustum.
   *
   * @param {Plane} [p0] - The first plane that encloses the frustum.
   * @param {Plane} [p1] - The second plane that encloses the frustum.
   * @param {Plane} [p2] - The third plane that encloses the frustum.
   * @param {Plane} [p3] - The fourth plane that encloses the frustum.
   * @param {Plane} [p4] - The fifth plane that encloses the frustum.
   * @param {Plane} [p5] - The sixth plane that encloses the frustum.
   */
  constructor(e = new mi(), t = new mi(), i = new mi(), s = new mi(), r = new mi(), o = new mi()) {
    this.planes = [e, t, i, s, r, o];
  }
  /**
   * Sets the frustum planes by copying the given planes.
   *
   * @param {Plane} [p0] - The first plane that encloses the frustum.
   * @param {Plane} [p1] - The second plane that encloses the frustum.
   * @param {Plane} [p2] - The third plane that encloses the frustum.
   * @param {Plane} [p3] - The fourth plane that encloses the frustum.
   * @param {Plane} [p4] - The fifth plane that encloses the frustum.
   * @param {Plane} [p5] - The sixth plane that encloses the frustum.
   * @return {Frustum} A reference to this frustum.
   */
  set(e, t, i, s, r, o) {
    const a = this.planes;
    return a[0].copy(e), a[1].copy(t), a[2].copy(i), a[3].copy(s), a[4].copy(r), a[5].copy(o), this;
  }
  /**
   * Copies the values of the given frustum to this instance.
   *
   * @param {Frustum} frustum - The frustum to copy.
   * @return {Frustum} A reference to this frustum.
   */
  copy(e) {
    const t = this.planes;
    for (let i = 0; i < 6; i++)
      t[i].copy(e.planes[i]);
    return this;
  }
  /**
   * Sets the frustum planes from the given projection matrix.
   *
   * @param {Matrix4} m - The projection matrix.
   * @param {(WebGLCoordinateSystem|WebGPUCoordinateSystem)} coordinateSystem - The coordinate system.
   * @param {boolean} [reversedDepth=false] - Whether to use a reversed depth.
   * @return {Frustum} A reference to this frustum.
   */
  setFromProjectionMatrix(e, t = Nn, i = !1) {
    const s = this.planes, r = e.elements, o = r[0], a = r[1], l = r[2], c = r[3], u = r[4], h = r[5], f = r[6], p = r[7], v = r[8], x = r[9], m = r[10], d = r[11], b = r[12], A = r[13], M = r[14], R = r[15];
    if (s[0].setComponents(c - o, p - u, d - v, R - b).normalize(), s[1].setComponents(c + o, p + u, d + v, R + b).normalize(), s[2].setComponents(c + a, p + h, d + x, R + A).normalize(), s[3].setComponents(c - a, p - h, d - x, R - A).normalize(), i)
      s[4].setComponents(l, f, m, M).normalize(), s[5].setComponents(c - l, p - f, d - m, R - M).normalize();
    else if (s[4].setComponents(c - l, p - f, d - m, R - M).normalize(), t === Nn)
      s[5].setComponents(c + l, p + f, d + m, R + M).normalize();
    else if (t === zo)
      s[5].setComponents(l, f, m, M).normalize();
    else
      throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: " + t);
    return this;
  }
  /**
   * Returns `true` if the 3D object's bounding sphere is intersecting this frustum.
   *
   * Note that the 3D object must have a geometry so that the bounding sphere can be calculated.
   *
   * @param {Object3D} object - The 3D object to test.
   * @return {boolean} Whether the 3D object's bounding sphere is intersecting this frustum or not.
   */
  intersectsObject(e) {
    if (e.boundingSphere !== void 0)
      e.boundingSphere === null && e.computeBoundingSphere(), Li.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);
    else {
      const t = e.geometry;
      t.boundingSphere === null && t.computeBoundingSphere(), Li.copy(t.boundingSphere).applyMatrix4(e.matrixWorld);
    }
    return this.intersectsSphere(Li);
  }
  /**
   * Returns `true` if the given sprite is intersecting this frustum.
   *
   * @param {Sprite} sprite - The sprite to test.
   * @return {boolean} Whether the sprite is intersecting this frustum or not.
   */
  intersectsSprite(e) {
    Li.center.set(0, 0, 0);
    const t = s0.distanceTo(e.center);
    return Li.radius = 0.7071067811865476 + t, Li.applyMatrix4(e.matrixWorld), this.intersectsSphere(Li);
  }
  /**
   * Returns `true` if the given bounding sphere is intersecting this frustum.
   *
   * @param {Sphere} sphere - The bounding sphere to test.
   * @return {boolean} Whether the bounding sphere is intersecting this frustum or not.
   */
  intersectsSphere(e) {
    const t = this.planes, i = e.center, s = -e.radius;
    for (let r = 0; r < 6; r++)
      if (t[r].distanceToPoint(i) < s)
        return !1;
    return !0;
  }
  /**
   * Returns `true` if the given bounding box is intersecting this frustum.
   *
   * @param {Box3} box - The bounding box to test.
   * @return {boolean} Whether the bounding box is intersecting this frustum or not.
   */
  intersectsBox(e) {
    const t = this.planes;
    for (let i = 0; i < 6; i++) {
      const s = t[i];
      if (so.x = s.normal.x > 0 ? e.max.x : e.min.x, so.y = s.normal.y > 0 ? e.max.y : e.min.y, so.z = s.normal.z > 0 ? e.max.z : e.min.z, s.distanceToPoint(so) < 0)
        return !1;
    }
    return !0;
  }
  /**
   * Returns `true` if the given point lies within the frustum.
   *
   * @param {Vector3} point - The point to test.
   * @return {boolean} Whether the point lies within this frustum or not.
   */
  containsPoint(e) {
    const t = this.planes;
    for (let i = 0; i < 6; i++)
      if (t[i].distanceToPoint(e) < 0)
        return !1;
    return !0;
  }
  /**
   * Returns a new frustum with copied values from this instance.
   *
   * @return {Frustum} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
}
class wc extends Ji {
  /**
   * Constructs a new line basic material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isLineBasicMaterial = !0, this.type = "LineBasicMaterial", this.color = new We(16777215), this.map = null, this.linewidth = 1, this.linecap = "round", this.linejoin = "round", this.fog = !0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.color.copy(e.color), this.map = e.map, this.linewidth = e.linewidth, this.linecap = e.linecap, this.linejoin = e.linejoin, this.fog = e.fog, this;
  }
}
const Vo = /* @__PURE__ */ new N(), ko = /* @__PURE__ */ new N(), ju = /* @__PURE__ */ new pt(), er = /* @__PURE__ */ new na(), ro = /* @__PURE__ */ new Nr(), Ga = /* @__PURE__ */ new N(), Ku = /* @__PURE__ */ new N();
class r0 extends Tt {
  /**
   * Constructs a new line.
   *
   * @param {BufferGeometry} [geometry] - The line geometry.
   * @param {Material|Array<Material>} [material] - The line material.
   */
  constructor(e = new Nt(), t = new wc()) {
    super(), this.isLine = !0, this.type = "Line", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.updateMorphTargets();
  }
  copy(e, t) {
    return super.copy(e, t), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
  }
  /**
   * Computes an array of distance values which are necessary for rendering dashed lines.
   * For each vertex in the geometry, the method calculates the cumulative length from the
   * current point to the very beginning of the line.
   *
   * @return {Line} A reference to this line.
   */
  computeLineDistances() {
    const e = this.geometry;
    if (e.index === null) {
      const t = e.attributes.position, i = [0];
      for (let s = 1, r = t.count; s < r; s++)
        Vo.fromBufferAttribute(t, s - 1), ko.fromBufferAttribute(t, s), i[s] = i[s - 1], i[s] += Vo.distanceTo(ko);
      e.setAttribute("lineDistance", new mt(i, 1));
    } else
      console.warn("THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");
    return this;
  }
  /**
   * Computes intersection points between a casted ray and this line.
   *
   * @param {Raycaster} raycaster - The raycaster.
   * @param {Array<Object>} intersects - The target array that holds the intersection points.
   */
  raycast(e, t) {
    const i = this.geometry, s = this.matrixWorld, r = e.params.Line.threshold, o = i.drawRange;
    if (i.boundingSphere === null && i.computeBoundingSphere(), ro.copy(i.boundingSphere), ro.applyMatrix4(s), ro.radius += r, e.ray.intersectsSphere(ro) === !1) return;
    ju.copy(s).invert(), er.copy(e.ray).applyMatrix4(ju);
    const a = r / ((this.scale.x + this.scale.y + this.scale.z) / 3), l = a * a, c = this.isLineSegments ? 2 : 1, u = i.index, f = i.attributes.position;
    if (u !== null) {
      const p = Math.max(0, o.start), v = Math.min(u.count, o.start + o.count);
      for (let x = p, m = v - 1; x < m; x += c) {
        const d = u.getX(x), b = u.getX(x + 1), A = oo(this, e, er, l, d, b, x);
        A && t.push(A);
      }
      if (this.isLineLoop) {
        const x = u.getX(v - 1), m = u.getX(p), d = oo(this, e, er, l, x, m, v - 1);
        d && t.push(d);
      }
    } else {
      const p = Math.max(0, o.start), v = Math.min(f.count, o.start + o.count);
      for (let x = p, m = v - 1; x < m; x += c) {
        const d = oo(this, e, er, l, x, x + 1, x);
        d && t.push(d);
      }
      if (this.isLineLoop) {
        const x = oo(this, e, er, l, v - 1, p, v - 1);
        x && t.push(x);
      }
    }
  }
  /**
   * Sets the values of {@link Line#morphTargetDictionary} and {@link Line#morphTargetInfluences}
   * to make sure existing morph targets can influence this 3D object.
   */
  updateMorphTargets() {
    const t = this.geometry.morphAttributes, i = Object.keys(t);
    if (i.length > 0) {
      const s = t[i[0]];
      if (s !== void 0) {
        this.morphTargetInfluences = [], this.morphTargetDictionary = {};
        for (let r = 0, o = s.length; r < o; r++) {
          const a = s[r].name || String(r);
          this.morphTargetInfluences.push(0), this.morphTargetDictionary[a] = r;
        }
      }
    }
  }
}
function oo(n, e, t, i, s, r, o) {
  const a = n.geometry.attributes.position;
  if (Vo.fromBufferAttribute(a, s), ko.fromBufferAttribute(a, r), t.distanceSqToSegment(Vo, ko, Ga, Ku) > i) return;
  Ga.applyMatrix4(n.matrixWorld);
  const c = e.ray.origin.distanceTo(Ga);
  if (!(c < e.near || c > e.far))
    return {
      distance: c,
      // What do we want? intersection point on the ray or on the segment??
      // point: raycaster.ray.at( distance ),
      point: Ku.clone().applyMatrix4(n.matrixWorld),
      index: o,
      face: null,
      faceIndex: null,
      barycoord: null,
      object: n
    };
}
const $u = /* @__PURE__ */ new N(), Zu = /* @__PURE__ */ new N();
class Sd extends r0 {
  /**
   * Constructs a new line segments.
   *
   * @param {BufferGeometry} [geometry] - The line geometry.
   * @param {Material|Array<Material>} [material] - The line material.
   */
  constructor(e, t) {
    super(e, t), this.isLineSegments = !0, this.type = "LineSegments";
  }
  computeLineDistances() {
    const e = this.geometry;
    if (e.index === null) {
      const t = e.attributes.position, i = [];
      for (let s = 0, r = t.count; s < r; s += 2)
        $u.fromBufferAttribute(t, s), Zu.fromBufferAttribute(t, s + 1), i[s] = s === 0 ? 0 : i[s - 1], i[s + 1] = i[s] + $u.distanceTo(Zu);
      e.setAttribute("lineDistance", new mt(i, 1));
    } else
      console.warn("THREE.LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");
    return this;
  }
}
class yd extends Ji {
  /**
   * Constructs a new points material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isPointsMaterial = !0, this.type = "PointsMaterial", this.color = new We(16777215), this.map = null, this.alphaMap = null, this.size = 1, this.sizeAttenuation = !0, this.fog = !0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.color.copy(e.color), this.map = e.map, this.alphaMap = e.alphaMap, this.size = e.size, this.sizeAttenuation = e.sizeAttenuation, this.fog = e.fog, this;
  }
}
const Ju = /* @__PURE__ */ new pt(), ec = /* @__PURE__ */ new na(), ao = /* @__PURE__ */ new Nr(), lo = /* @__PURE__ */ new N();
class o0 extends Tt {
  /**
   * Constructs a new point cloud.
   *
   * @param {BufferGeometry} [geometry] - The points geometry.
   * @param {Material|Array<Material>} [material] - The points material.
   */
  constructor(e = new Nt(), t = new yd()) {
    super(), this.isPoints = !0, this.type = "Points", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.updateMorphTargets();
  }
  copy(e, t) {
    return super.copy(e, t), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
  }
  /**
   * Computes intersection points between a casted ray and this point cloud.
   *
   * @param {Raycaster} raycaster - The raycaster.
   * @param {Array<Object>} intersects - The target array that holds the intersection points.
   */
  raycast(e, t) {
    const i = this.geometry, s = this.matrixWorld, r = e.params.Points.threshold, o = i.drawRange;
    if (i.boundingSphere === null && i.computeBoundingSphere(), ao.copy(i.boundingSphere), ao.applyMatrix4(s), ao.radius += r, e.ray.intersectsSphere(ao) === !1) return;
    Ju.copy(s).invert(), ec.copy(e.ray).applyMatrix4(Ju);
    const a = r / ((this.scale.x + this.scale.y + this.scale.z) / 3), l = a * a, c = i.index, h = i.attributes.position;
    if (c !== null) {
      const f = Math.max(0, o.start), p = Math.min(c.count, o.start + o.count);
      for (let v = f, x = p; v < x; v++) {
        const m = c.getX(v);
        lo.fromBufferAttribute(h, m), Qu(lo, m, l, s, e, t, this);
      }
    } else {
      const f = Math.max(0, o.start), p = Math.min(h.count, o.start + o.count);
      for (let v = f, x = p; v < x; v++)
        lo.fromBufferAttribute(h, v), Qu(lo, v, l, s, e, t, this);
    }
  }
  /**
   * Sets the values of {@link Points#morphTargetDictionary} and {@link Points#morphTargetInfluences}
   * to make sure existing morph targets can influence this 3D object.
   */
  updateMorphTargets() {
    const t = this.geometry.morphAttributes, i = Object.keys(t);
    if (i.length > 0) {
      const s = t[i[0]];
      if (s !== void 0) {
        this.morphTargetInfluences = [], this.morphTargetDictionary = {};
        for (let r = 0, o = s.length; r < o; r++) {
          const a = s[r].name || String(r);
          this.morphTargetInfluences.push(0), this.morphTargetDictionary[a] = r;
        }
      }
    }
  }
}
function Qu(n, e, t, i, s, r, o) {
  const a = ec.distanceSqToPoint(n);
  if (a < t) {
    const l = new N();
    ec.closestPointToPoint(n, l), l.applyMatrix4(i);
    const c = s.ray.origin.distanceTo(l);
    if (c < s.near || c > s.far) return;
    r.push({
      distance: c,
      distanceToRay: Math.sqrt(a),
      point: l,
      index: e,
      face: null,
      faceIndex: null,
      barycoord: null,
      object: o
    });
  }
}
class Ed extends $t {
  /**
   * Constructs a new depth texture.
   *
   * @param {number} width - The width of the texture.
   * @param {number} height - The height of the texture.
   * @param {number} [type=UnsignedIntType] - The texture type.
   * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
   * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
   * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
   * @param {number} [magFilter=LinearFilter] - The mag filter value.
   * @param {number} [minFilter=LinearFilter] - The min filter value.
   * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
   * @param {number} [format=DepthFormat] - The texture format.
   * @param {number} [depth=1] - The depth of the texture.
   */
  constructor(e, t, i = Xi, s, r, o, a = yn, l = yn, c, u = Ar, h = 1) {
    if (u !== Ar && u !== wr)
      throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");
    const f = { width: e, height: t, depth: h };
    super(f, s, r, o, a, l, u, i, c), this.isDepthTexture = !0, this.flipY = !1, this.generateMipmaps = !1, this.compareFunction = null;
  }
  copy(e) {
    return super.copy(e), this.source = new bc(Object.assign({}, e.image)), this.compareFunction = e.compareFunction, this;
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return this.compareFunction !== null && (t.compareFunction = this.compareFunction), t;
  }
}
class Td extends $t {
  /**
   * Creates a new raw texture.
   *
   * @param {?(WebGLTexture|GPUTexture)} [sourceTexture=null] - The external texture.
   */
  constructor(e = null) {
    super(), this.sourceTexture = e, this.isExternalTexture = !0;
  }
  copy(e) {
    return super.copy(e), this.sourceTexture = e.sourceTexture, this;
  }
}
class Rc extends Nt {
  /**
   * Constructs a new capsule geometry.
   *
   * @param {number} [radius=1] - Radius of the capsule.
   * @param {number} [height=1] - Height of the middle section.
   * @param {number} [capSegments=4] - Number of curve segments used to build each cap.
   * @param {number} [radialSegments=8] - Number of segmented faces around the circumference of the capsule. Must be an integer >= 3.
   * @param {number} [heightSegments=1] - Number of rows of faces along the height of the middle section. Must be an integer >= 1.
   */
  constructor(e = 1, t = 1, i = 4, s = 8, r = 1) {
    super(), this.type = "CapsuleGeometry", this.parameters = {
      radius: e,
      height: t,
      capSegments: i,
      radialSegments: s,
      heightSegments: r
    }, t = Math.max(0, t), i = Math.max(1, Math.floor(i)), s = Math.max(3, Math.floor(s)), r = Math.max(1, Math.floor(r));
    const o = [], a = [], l = [], c = [], u = t / 2, h = Math.PI / 2 * e, f = t, p = 2 * h + f, v = i * 2 + r, x = s + 1, m = new N(), d = new N();
    for (let b = 0; b <= v; b++) {
      let A = 0, M = 0, R = 0, w = 0;
      if (b <= i) {
        const y = b / i, S = y * Math.PI / 2;
        M = -u - e * Math.cos(S), R = e * Math.sin(S), w = -e * Math.cos(S), A = y * h;
      } else if (b <= i + r) {
        const y = (b - i) / r;
        M = -u + y * t, R = e, w = 0, A = h + y * f;
      } else {
        const y = (b - i - r) / i, S = y * Math.PI / 2;
        M = u + e * Math.sin(S), R = e * Math.cos(S), w = e * Math.sin(S), A = h + f + y * h;
      }
      const D = Math.max(0, Math.min(1, A / p));
      let U = 0;
      b === 0 ? U = 0.5 / s : b === v && (U = -0.5 / s);
      for (let y = 0; y <= s; y++) {
        const S = y / s, P = S * Math.PI * 2, L = Math.sin(P), V = Math.cos(P);
        d.x = -R * V, d.y = M, d.z = R * L, a.push(d.x, d.y, d.z), m.set(
          -R * V,
          w,
          R * L
        ), m.normalize(), l.push(m.x, m.y, m.z), c.push(S + U, D);
      }
      if (b > 0) {
        const y = (b - 1) * x;
        for (let S = 0; S < s; S++) {
          const P = y + S, L = y + S + 1, V = b * x + S, Z = b * x + S + 1;
          o.push(P, L, V), o.push(L, Z, V);
        }
      }
    }
    this.setIndex(o), this.setAttribute("position", new mt(a, 3)), this.setAttribute("normal", new mt(l, 3)), this.setAttribute("uv", new mt(c, 2));
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {CapsuleGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Rc(e.radius, e.height, e.capSegments, e.radialSegments, e.heightSegments);
  }
}
class Cc extends Nt {
  /**
   * Constructs a new polyhedron geometry.
   *
   * @param {Array<number>} [vertices] - A flat array of vertices describing the base shape.
   * @param {Array<number>} [indices] - A flat array of indices describing the base shape.
   * @param {number} [radius=1] - The radius of the shape.
   * @param {number} [detail=0] - How many levels to subdivide the geometry. The more detail, the smoother the shape.
   */
  constructor(e = [], t = [], i = 1, s = 0) {
    super(), this.type = "PolyhedronGeometry", this.parameters = {
      vertices: e,
      indices: t,
      radius: i,
      detail: s
    };
    const r = [], o = [];
    a(s), c(i), u(), this.setAttribute("position", new mt(r, 3)), this.setAttribute("normal", new mt(r.slice(), 3)), this.setAttribute("uv", new mt(o, 2)), s === 0 ? this.computeVertexNormals() : this.normalizeNormals();
    function a(b) {
      const A = new N(), M = new N(), R = new N();
      for (let w = 0; w < t.length; w += 3)
        p(t[w + 0], A), p(t[w + 1], M), p(t[w + 2], R), l(A, M, R, b);
    }
    function l(b, A, M, R) {
      const w = R + 1, D = [];
      for (let U = 0; U <= w; U++) {
        D[U] = [];
        const y = b.clone().lerp(M, U / w), S = A.clone().lerp(M, U / w), P = w - U;
        for (let L = 0; L <= P; L++)
          L === 0 && U === w ? D[U][L] = y : D[U][L] = y.clone().lerp(S, L / P);
      }
      for (let U = 0; U < w; U++)
        for (let y = 0; y < 2 * (w - U) - 1; y++) {
          const S = Math.floor(y / 2);
          y % 2 === 0 ? (f(D[U][S + 1]), f(D[U + 1][S]), f(D[U][S])) : (f(D[U][S + 1]), f(D[U + 1][S + 1]), f(D[U + 1][S]));
        }
    }
    function c(b) {
      const A = new N();
      for (let M = 0; M < r.length; M += 3)
        A.x = r[M + 0], A.y = r[M + 1], A.z = r[M + 2], A.normalize().multiplyScalar(b), r[M + 0] = A.x, r[M + 1] = A.y, r[M + 2] = A.z;
    }
    function u() {
      const b = new N();
      for (let A = 0; A < r.length; A += 3) {
        b.x = r[A + 0], b.y = r[A + 1], b.z = r[A + 2];
        const M = m(b) / 2 / Math.PI + 0.5, R = d(b) / Math.PI + 0.5;
        o.push(M, 1 - R);
      }
      v(), h();
    }
    function h() {
      for (let b = 0; b < o.length; b += 6) {
        const A = o[b + 0], M = o[b + 2], R = o[b + 4], w = Math.max(A, M, R), D = Math.min(A, M, R);
        w > 0.9 && D < 0.1 && (A < 0.2 && (o[b + 0] += 1), M < 0.2 && (o[b + 2] += 1), R < 0.2 && (o[b + 4] += 1));
      }
    }
    function f(b) {
      r.push(b.x, b.y, b.z);
    }
    function p(b, A) {
      const M = b * 3;
      A.x = e[M + 0], A.y = e[M + 1], A.z = e[M + 2];
    }
    function v() {
      const b = new N(), A = new N(), M = new N(), R = new N(), w = new Ve(), D = new Ve(), U = new Ve();
      for (let y = 0, S = 0; y < r.length; y += 9, S += 6) {
        b.set(r[y + 0], r[y + 1], r[y + 2]), A.set(r[y + 3], r[y + 4], r[y + 5]), M.set(r[y + 6], r[y + 7], r[y + 8]), w.set(o[S + 0], o[S + 1]), D.set(o[S + 2], o[S + 3]), U.set(o[S + 4], o[S + 5]), R.copy(b).add(A).add(M).divideScalar(3);
        const P = m(R);
        x(w, S + 0, b, P), x(D, S + 2, A, P), x(U, S + 4, M, P);
      }
    }
    function x(b, A, M, R) {
      R < 0 && b.x === 1 && (o[A] = b.x - 1), M.x === 0 && M.z === 0 && (o[A] = R / 2 / Math.PI + 0.5);
    }
    function m(b) {
      return Math.atan2(b.z, -b.x);
    }
    function d(b) {
      return Math.atan2(-b.y, Math.sqrt(b.x * b.x + b.z * b.z));
    }
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {PolyhedronGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Cc(e.vertices, e.indices, e.radius, e.details);
  }
}
const co = /* @__PURE__ */ new N(), uo = /* @__PURE__ */ new N(), Wa = /* @__PURE__ */ new N(), ho = /* @__PURE__ */ new fn();
class a0 extends Nt {
  /**
   * Constructs a new edges geometry.
   *
   * @param {?BufferGeometry} [geometry=null] - The geometry.
   * @param {number} [thresholdAngle=1] - An edge is only rendered if the angle (in degrees)
   * between the face normals of the adjoining faces exceeds this value.
   */
  constructor(e = null, t = 1) {
    if (super(), this.type = "EdgesGeometry", this.parameters = {
      geometry: e,
      thresholdAngle: t
    }, e !== null) {
      const s = Math.pow(10, 4), r = Math.cos(dr * t), o = e.getIndex(), a = e.getAttribute("position"), l = o ? o.count : a.count, c = [0, 0, 0], u = ["a", "b", "c"], h = new Array(3), f = {}, p = [];
      for (let v = 0; v < l; v += 3) {
        o ? (c[0] = o.getX(v), c[1] = o.getX(v + 1), c[2] = o.getX(v + 2)) : (c[0] = v, c[1] = v + 1, c[2] = v + 2);
        const { a: x, b: m, c: d } = ho;
        if (x.fromBufferAttribute(a, c[0]), m.fromBufferAttribute(a, c[1]), d.fromBufferAttribute(a, c[2]), ho.getNormal(Wa), h[0] = `${Math.round(x.x * s)},${Math.round(x.y * s)},${Math.round(x.z * s)}`, h[1] = `${Math.round(m.x * s)},${Math.round(m.y * s)},${Math.round(m.z * s)}`, h[2] = `${Math.round(d.x * s)},${Math.round(d.y * s)},${Math.round(d.z * s)}`, !(h[0] === h[1] || h[1] === h[2] || h[2] === h[0]))
          for (let b = 0; b < 3; b++) {
            const A = (b + 1) % 3, M = h[b], R = h[A], w = ho[u[b]], D = ho[u[A]], U = `${M}_${R}`, y = `${R}_${M}`;
            y in f && f[y] ? (Wa.dot(f[y].normal) <= r && (p.push(w.x, w.y, w.z), p.push(D.x, D.y, D.z)), f[y] = null) : U in f || (f[U] = {
              index0: c[b],
              index1: c[A],
              normal: Wa.clone()
            });
          }
      }
      for (const v in f)
        if (f[v]) {
          const { index0: x, index1: m } = f[v];
          co.fromBufferAttribute(a, x), uo.fromBufferAttribute(a, m), p.push(co.x, co.y, co.z), p.push(uo.x, uo.y, uo.z);
        }
      this.setAttribute("position", new mt(p, 3));
    }
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
}
class Pc extends Cc {
  /**
   * Constructs a new icosahedron geometry.
   *
   * @param {number} [radius=1] - Radius of the icosahedron.
   * @param {number} [detail=0] - Setting this to a value greater than `0` adds vertices making it no longer a icosahedron.
   */
  constructor(e = 1, t = 0) {
    const i = (1 + Math.sqrt(5)) / 2, s = [
      -1,
      i,
      0,
      1,
      i,
      0,
      -1,
      -i,
      0,
      1,
      -i,
      0,
      0,
      -1,
      i,
      0,
      1,
      i,
      0,
      -1,
      -i,
      0,
      1,
      -i,
      i,
      0,
      -1,
      i,
      0,
      1,
      -i,
      0,
      -1,
      -i,
      0,
      1
    ], r = [
      0,
      11,
      5,
      0,
      5,
      1,
      0,
      1,
      7,
      0,
      7,
      10,
      0,
      10,
      11,
      1,
      5,
      9,
      5,
      11,
      4,
      11,
      10,
      2,
      10,
      7,
      6,
      7,
      1,
      8,
      3,
      9,
      4,
      3,
      4,
      2,
      3,
      2,
      6,
      3,
      6,
      8,
      3,
      8,
      9,
      4,
      9,
      5,
      2,
      4,
      11,
      6,
      2,
      10,
      8,
      6,
      7,
      9,
      8,
      1
    ];
    super(s, r, e, t), this.type = "IcosahedronGeometry", this.parameters = {
      radius: e,
      detail: t
    };
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {IcosahedronGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Pc(e.radius, e.detail);
  }
}
class zs extends Nt {
  /**
   * Constructs a new plane geometry.
   *
   * @param {number} [width=1] - The width along the X axis.
   * @param {number} [height=1] - The height along the Y axis
   * @param {number} [widthSegments=1] - The number of segments along the X axis.
   * @param {number} [heightSegments=1] - The number of segments along the Y axis.
   */
  constructor(e = 1, t = 1, i = 1, s = 1) {
    super(), this.type = "PlaneGeometry", this.parameters = {
      width: e,
      height: t,
      widthSegments: i,
      heightSegments: s
    };
    const r = e / 2, o = t / 2, a = Math.floor(i), l = Math.floor(s), c = a + 1, u = l + 1, h = e / a, f = t / l, p = [], v = [], x = [], m = [];
    for (let d = 0; d < u; d++) {
      const b = d * f - o;
      for (let A = 0; A < c; A++) {
        const M = A * h - r;
        v.push(M, -b, 0), x.push(0, 0, 1), m.push(A / a), m.push(1 - d / l);
      }
    }
    for (let d = 0; d < l; d++)
      for (let b = 0; b < a; b++) {
        const A = b + c * d, M = b + c * (d + 1), R = b + 1 + c * (d + 1), w = b + 1 + c * d;
        p.push(A, M, w), p.push(M, R, w);
      }
    this.setIndex(p), this.setAttribute("position", new mt(v, 3)), this.setAttribute("normal", new mt(x, 3)), this.setAttribute("uv", new mt(m, 2));
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {PlaneGeometry} A new instance.
   */
  static fromJSON(e) {
    return new zs(e.width, e.height, e.widthSegments, e.heightSegments);
  }
}
class ys extends Nt {
  /**
   * Constructs a new sphere geometry.
   *
   * @param {number} [radius=1] - The sphere radius.
   * @param {number} [widthSegments=32] - The number of horizontal segments. Minimum value is `3`.
   * @param {number} [heightSegments=16] - The number of vertical segments. Minimum value is `2`.
   * @param {number} [phiStart=0] - The horizontal starting angle in radians.
   * @param {number} [phiLength=Math.PI*2] - The horizontal sweep angle size.
   * @param {number} [thetaStart=0] - The vertical starting angle in radians.
   * @param {number} [thetaLength=Math.PI] - The vertical sweep angle size.
   */
  constructor(e = 1, t = 32, i = 16, s = 0, r = Math.PI * 2, o = 0, a = Math.PI) {
    super(), this.type = "SphereGeometry", this.parameters = {
      radius: e,
      widthSegments: t,
      heightSegments: i,
      phiStart: s,
      phiLength: r,
      thetaStart: o,
      thetaLength: a
    }, t = Math.max(3, Math.floor(t)), i = Math.max(2, Math.floor(i));
    const l = Math.min(o + a, Math.PI);
    let c = 0;
    const u = [], h = new N(), f = new N(), p = [], v = [], x = [], m = [];
    for (let d = 0; d <= i; d++) {
      const b = [], A = d / i;
      let M = 0;
      d === 0 && o === 0 ? M = 0.5 / t : d === i && l === Math.PI && (M = -0.5 / t);
      for (let R = 0; R <= t; R++) {
        const w = R / t;
        h.x = -e * Math.cos(s + w * r) * Math.sin(o + A * a), h.y = e * Math.cos(o + A * a), h.z = e * Math.sin(s + w * r) * Math.sin(o + A * a), v.push(h.x, h.y, h.z), f.copy(h).normalize(), x.push(f.x, f.y, f.z), m.push(w + M, 1 - A), b.push(c++);
      }
      u.push(b);
    }
    for (let d = 0; d < i; d++)
      for (let b = 0; b < t; b++) {
        const A = u[d][b + 1], M = u[d][b], R = u[d + 1][b], w = u[d + 1][b + 1];
        (d !== 0 || o > 0) && p.push(A, M, w), (d !== i - 1 || l < Math.PI) && p.push(M, R, w);
      }
    this.setIndex(p), this.setAttribute("position", new mt(v, 3)), this.setAttribute("normal", new mt(x, 3)), this.setAttribute("uv", new mt(m, 2));
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {SphereGeometry} A new instance.
   */
  static fromJSON(e) {
    return new ys(e.radius, e.widthSegments, e.heightSegments, e.phiStart, e.phiLength, e.thetaStart, e.thetaLength);
  }
}
class Es extends Nt {
  /**
   * Constructs a new torus geometry.
   *
   * @param {number} [radius=1] - Radius of the torus, from the center of the torus to the center of the tube.
   * @param {number} [tube=0.4] - Radius of the tube. Must be smaller than `radius`.
   * @param {number} [radialSegments=12] - The number of radial segments.
   * @param {number} [tubularSegments=48] - The number of tubular segments.
   * @param {number} [arc=Math.PI*2] - Central angle in radians.
   */
  constructor(e = 1, t = 0.4, i = 12, s = 48, r = Math.PI * 2) {
    super(), this.type = "TorusGeometry", this.parameters = {
      radius: e,
      tube: t,
      radialSegments: i,
      tubularSegments: s,
      arc: r
    }, i = Math.floor(i), s = Math.floor(s);
    const o = [], a = [], l = [], c = [], u = new N(), h = new N(), f = new N();
    for (let p = 0; p <= i; p++)
      for (let v = 0; v <= s; v++) {
        const x = v / s * r, m = p / i * Math.PI * 2;
        h.x = (e + t * Math.cos(m)) * Math.cos(x), h.y = (e + t * Math.cos(m)) * Math.sin(x), h.z = t * Math.sin(m), a.push(h.x, h.y, h.z), u.x = e * Math.cos(x), u.y = e * Math.sin(x), f.subVectors(h, u).normalize(), l.push(f.x, f.y, f.z), c.push(v / s), c.push(p / i);
      }
    for (let p = 1; p <= i; p++)
      for (let v = 1; v <= s; v++) {
        const x = (s + 1) * p + v - 1, m = (s + 1) * (p - 1) + v - 1, d = (s + 1) * (p - 1) + v, b = (s + 1) * p + v;
        o.push(x, m, b), o.push(m, d, b);
      }
    this.setIndex(o), this.setAttribute("position", new mt(a, 3)), this.setAttribute("normal", new mt(l, 3)), this.setAttribute("uv", new mt(c, 2));
  }
  copy(e) {
    return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
  }
  /**
   * Factory method for creating an instance of this class from the given
   * JSON object.
   *
   * @param {Object} data - A JSON object representing the serialized geometry.
   * @return {TorusGeometry} A new instance.
   */
  static fromJSON(e) {
    return new Es(e.radius, e.tube, e.radialSegments, e.tubularSegments, e.arc);
  }
}
class bo extends Ji {
  /**
   * Constructs a new mesh standard material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshStandardMaterial = !0, this.type = "MeshStandardMaterial", this.defines = { STANDARD: "" }, this.color = new We(16777215), this.roughness = 1, this.metalness = 0, this.map = null, this.lightMap = null, this.lightMapIntensity = 1, this.aoMap = null, this.aoMapIntensity = 1, this.emissive = new We(0), this.emissiveIntensity = 1, this.emissiveMap = null, this.bumpMap = null, this.bumpScale = 1, this.normalMap = null, this.normalMapType = ud, this.normalScale = new Ve(1, 1), this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.roughnessMap = null, this.metalnessMap = null, this.alphaMap = null, this.envMap = null, this.envMapRotation = new zn(), this.envMapIntensity = 1, this.wireframe = !1, this.wireframeLinewidth = 1, this.wireframeLinecap = "round", this.wireframeLinejoin = "round", this.flatShading = !1, this.fog = !0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.defines = { STANDARD: "" }, this.color.copy(e.color), this.roughness = e.roughness, this.metalness = e.metalness, this.map = e.map, this.lightMap = e.lightMap, this.lightMapIntensity = e.lightMapIntensity, this.aoMap = e.aoMap, this.aoMapIntensity = e.aoMapIntensity, this.emissive.copy(e.emissive), this.emissiveMap = e.emissiveMap, this.emissiveIntensity = e.emissiveIntensity, this.bumpMap = e.bumpMap, this.bumpScale = e.bumpScale, this.normalMap = e.normalMap, this.normalMapType = e.normalMapType, this.normalScale.copy(e.normalScale), this.displacementMap = e.displacementMap, this.displacementScale = e.displacementScale, this.displacementBias = e.displacementBias, this.roughnessMap = e.roughnessMap, this.metalnessMap = e.metalnessMap, this.alphaMap = e.alphaMap, this.envMap = e.envMap, this.envMapRotation.copy(e.envMapRotation), this.envMapIntensity = e.envMapIntensity, this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this.wireframeLinecap = e.wireframeLinecap, this.wireframeLinejoin = e.wireframeLinejoin, this.flatShading = e.flatShading, this.fog = e.fog, this;
  }
}
class eh extends bo {
  /**
   * Constructs a new mesh physical material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshPhysicalMaterial = !0, this.defines = {
      STANDARD: "",
      PHYSICAL: ""
    }, this.type = "MeshPhysicalMaterial", this.anisotropyRotation = 0, this.anisotropyMap = null, this.clearcoatMap = null, this.clearcoatRoughness = 0, this.clearcoatRoughnessMap = null, this.clearcoatNormalScale = new Ve(1, 1), this.clearcoatNormalMap = null, this.ior = 1.5, Object.defineProperty(this, "reflectivity", {
      get: function() {
        return je(2.5 * (this.ior - 1) / (this.ior + 1), 0, 1);
      },
      set: function(t) {
        this.ior = (1 + 0.4 * t) / (1 - 0.4 * t);
      }
    }), this.iridescenceMap = null, this.iridescenceIOR = 1.3, this.iridescenceThicknessRange = [100, 400], this.iridescenceThicknessMap = null, this.sheenColor = new We(0), this.sheenColorMap = null, this.sheenRoughness = 1, this.sheenRoughnessMap = null, this.transmissionMap = null, this.thickness = 0, this.thicknessMap = null, this.attenuationDistance = 1 / 0, this.attenuationColor = new We(1, 1, 1), this.specularIntensity = 1, this.specularIntensityMap = null, this.specularColor = new We(1, 1, 1), this.specularColorMap = null, this._anisotropy = 0, this._clearcoat = 0, this._dispersion = 0, this._iridescence = 0, this._sheen = 0, this._transmission = 0, this.setValues(e);
  }
  /**
   * The anisotropy strength.
   *
   * @type {number}
   * @default 0
   */
  get anisotropy() {
    return this._anisotropy;
  }
  set anisotropy(e) {
    this._anisotropy > 0 != e > 0 && this.version++, this._anisotropy = e;
  }
  /**
   * Represents the intensity of the clear coat layer, from `0.0` to `1.0`. Use
   * clear coat related properties to enable multilayer materials that have a
   * thin translucent layer over the base layer.
   *
   * @type {number}
   * @default 0
   */
  get clearcoat() {
    return this._clearcoat;
  }
  set clearcoat(e) {
    this._clearcoat > 0 != e > 0 && this.version++, this._clearcoat = e;
  }
  /**
   * The intensity of the iridescence layer, simulating RGB color shift based on the angle between
   * the surface and the viewer, from `0.0` to `1.0`.
   *
   * @type {number}
   * @default 0
   */
  get iridescence() {
    return this._iridescence;
  }
  set iridescence(e) {
    this._iridescence > 0 != e > 0 && this.version++, this._iridescence = e;
  }
  /**
   * Defines the strength of the angular separation of colors (chromatic aberration) transmitting
   * through a relatively clear volume. Any value zero or larger is valid, the typical range of
   * realistic values is `[0, 1]`. This property can be only be used with transmissive objects.
   *
   * @type {number}
   * @default 0
   */
  get dispersion() {
    return this._dispersion;
  }
  set dispersion(e) {
    this._dispersion > 0 != e > 0 && this.version++, this._dispersion = e;
  }
  /**
   * The intensity of the sheen layer, from `0.0` to `1.0`.
   *
   * @type {number}
   * @default 0
   */
  get sheen() {
    return this._sheen;
  }
  set sheen(e) {
    this._sheen > 0 != e > 0 && this.version++, this._sheen = e;
  }
  /**
   * Degree of transmission (or optical transparency), from `0.0` to `1.0`.
   *
   * Thin, transparent or semitransparent, plastic or glass materials remain
   * largely reflective even if they are fully transmissive. The transmission
   * property can be used to model these materials.
   *
   * When transmission is non-zero, `opacity` should be  set to `1`.
   *
   * @type {number}
   * @default 0
   */
  get transmission() {
    return this._transmission;
  }
  set transmission(e) {
    this._transmission > 0 != e > 0 && this.version++, this._transmission = e;
  }
  copy(e) {
    return super.copy(e), this.defines = {
      STANDARD: "",
      PHYSICAL: ""
    }, this.anisotropy = e.anisotropy, this.anisotropyRotation = e.anisotropyRotation, this.anisotropyMap = e.anisotropyMap, this.clearcoat = e.clearcoat, this.clearcoatMap = e.clearcoatMap, this.clearcoatRoughness = e.clearcoatRoughness, this.clearcoatRoughnessMap = e.clearcoatRoughnessMap, this.clearcoatNormalMap = e.clearcoatNormalMap, this.clearcoatNormalScale.copy(e.clearcoatNormalScale), this.dispersion = e.dispersion, this.ior = e.ior, this.iridescence = e.iridescence, this.iridescenceMap = e.iridescenceMap, this.iridescenceIOR = e.iridescenceIOR, this.iridescenceThicknessRange = [...e.iridescenceThicknessRange], this.iridescenceThicknessMap = e.iridescenceThicknessMap, this.sheen = e.sheen, this.sheenColor.copy(e.sheenColor), this.sheenColorMap = e.sheenColorMap, this.sheenRoughness = e.sheenRoughness, this.sheenRoughnessMap = e.sheenRoughnessMap, this.transmission = e.transmission, this.transmissionMap = e.transmissionMap, this.thickness = e.thickness, this.thicknessMap = e.thicknessMap, this.attenuationDistance = e.attenuationDistance, this.attenuationColor.copy(e.attenuationColor), this.specularIntensity = e.specularIntensity, this.specularIntensityMap = e.specularIntensityMap, this.specularColor.copy(e.specularColor), this.specularColorMap = e.specularColorMap, this;
  }
}
class l0 extends Ji {
  /**
   * Constructs a new mesh depth material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshDepthMaterial = !0, this.type = "MeshDepthMaterial", this.depthPacking = gg, this.map = null, this.alphaMap = null, this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.wireframe = !1, this.wireframeLinewidth = 1, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.depthPacking = e.depthPacking, this.map = e.map, this.alphaMap = e.alphaMap, this.displacementMap = e.displacementMap, this.displacementScale = e.displacementScale, this.displacementBias = e.displacementBias, this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this;
  }
}
class c0 extends Ji {
  /**
   * Constructs a new mesh distance material.
   *
   * @param {Object} [parameters] - An object with one or more properties
   * defining the material's appearance. Any property of the material
   * (including any property from inherited materials) can be passed
   * in here. Color values can be passed any type of value accepted
   * by {@link Color#set}.
   */
  constructor(e) {
    super(), this.isMeshDistanceMaterial = !0, this.type = "MeshDistanceMaterial", this.map = null, this.alphaMap = null, this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.setValues(e);
  }
  copy(e) {
    return super.copy(e), this.map = e.map, this.alphaMap = e.alphaMap, this.displacementMap = e.displacementMap, this.displacementScale = e.displacementScale, this.displacementBias = e.displacementBias, this;
  }
}
class Dc extends Tt {
  /**
   * Constructs a new light.
   *
   * @param {(number|Color|string)} [color=0xffffff] - The light's color.
   * @param {number} [intensity=1] - The light's strength/intensity.
   */
  constructor(e, t = 1) {
    super(), this.isLight = !0, this.type = "Light", this.color = new We(e), this.intensity = t;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   */
  dispose() {
  }
  copy(e, t) {
    return super.copy(e, t), this.color.copy(e.color), this.intensity = e.intensity, this;
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return t.object.color = this.color.getHex(), t.object.intensity = this.intensity, this.groundColor !== void 0 && (t.object.groundColor = this.groundColor.getHex()), this.distance !== void 0 && (t.object.distance = this.distance), this.angle !== void 0 && (t.object.angle = this.angle), this.decay !== void 0 && (t.object.decay = this.decay), this.penumbra !== void 0 && (t.object.penumbra = this.penumbra), this.shadow !== void 0 && (t.object.shadow = this.shadow.toJSON()), this.target !== void 0 && (t.object.target = this.target.uuid), t;
  }
}
class u0 extends Dc {
  /**
   * Constructs a new hemisphere light.
   *
   * @param {(number|Color|string)} [skyColor=0xffffff] - The light's sky color.
   * @param {(number|Color|string)} [groundColor=0xffffff] - The light's ground color.
   * @param {number} [intensity=1] - The light's strength/intensity.
   */
  constructor(e, t, i) {
    super(e, i), this.isHemisphereLight = !0, this.type = "HemisphereLight", this.position.copy(Tt.DEFAULT_UP), this.updateMatrix(), this.groundColor = new We(t);
  }
  copy(e, t) {
    return super.copy(e, t), this.groundColor.copy(e.groundColor), this;
  }
}
const Xa = /* @__PURE__ */ new pt(), th = /* @__PURE__ */ new N(), nh = /* @__PURE__ */ new N();
class bd {
  /**
   * Constructs a new light shadow.
   *
   * @param {Camera} camera - The light's view of the world.
   */
  constructor(e) {
    this.camera = e, this.intensity = 1, this.bias = 0, this.normalBias = 0, this.radius = 1, this.blurSamples = 8, this.mapSize = new Ve(512, 512), this.mapType = Bn, this.map = null, this.mapPass = null, this.matrix = new pt(), this.autoUpdate = !0, this.needsUpdate = !1, this._frustum = new Ac(), this._frameExtents = new Ve(1, 1), this._viewportCount = 1, this._viewports = [
      new lt(0, 0, 1, 1)
    ];
  }
  /**
   * Used internally by the renderer to get the number of viewports that need
   * to be rendered for this shadow.
   *
   * @return {number} The viewport count.
   */
  getViewportCount() {
    return this._viewportCount;
  }
  /**
   * Gets the shadow cameras frustum. Used internally by the renderer to cull objects.
   *
   * @return {Frustum} The shadow camera frustum.
   */
  getFrustum() {
    return this._frustum;
  }
  /**
   * Update the matrices for the camera and shadow, used internally by the renderer.
   *
   * @param {Light} light - The light for which the shadow is being rendered.
   */
  updateMatrices(e) {
    const t = this.camera, i = this.matrix;
    th.setFromMatrixPosition(e.matrixWorld), t.position.copy(th), nh.setFromMatrixPosition(e.target.matrixWorld), t.lookAt(nh), t.updateMatrixWorld(), Xa.multiplyMatrices(t.projectionMatrix, t.matrixWorldInverse), this._frustum.setFromProjectionMatrix(Xa, t.coordinateSystem, t.reversedDepth), t.reversedDepth ? i.set(
      0.5,
      0,
      0,
      0.5,
      0,
      0.5,
      0,
      0.5,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ) : i.set(
      0.5,
      0,
      0,
      0.5,
      0,
      0.5,
      0,
      0.5,
      0,
      0,
      0.5,
      0.5,
      0,
      0,
      0,
      1
    ), i.multiply(Xa);
  }
  /**
   * Returns a viewport definition for the given viewport index.
   *
   * @param {number} viewportIndex - The viewport index.
   * @return {Vector4} The viewport.
   */
  getViewport(e) {
    return this._viewports[e];
  }
  /**
   * Returns the frame extends.
   *
   * @return {Vector2} The frame extends.
   */
  getFrameExtents() {
    return this._frameExtents;
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   */
  dispose() {
    this.map && this.map.dispose(), this.mapPass && this.mapPass.dispose();
  }
  /**
   * Copies the values of the given light shadow instance to this instance.
   *
   * @param {LightShadow} source - The light shadow to copy.
   * @return {LightShadow} A reference to this light shadow instance.
   */
  copy(e) {
    return this.camera = e.camera.clone(), this.intensity = e.intensity, this.bias = e.bias, this.radius = e.radius, this.autoUpdate = e.autoUpdate, this.needsUpdate = e.needsUpdate, this.normalBias = e.normalBias, this.blurSamples = e.blurSamples, this.mapSize.copy(e.mapSize), this;
  }
  /**
   * Returns a new light shadow instance with copied values from this instance.
   *
   * @return {LightShadow} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
  /**
   * Serializes the light shadow into JSON.
   *
   * @return {Object} A JSON object representing the serialized light shadow.
   * @see {@link ObjectLoader#parse}
   */
  toJSON() {
    const e = {};
    return this.intensity !== 1 && (e.intensity = this.intensity), this.bias !== 0 && (e.bias = this.bias), this.normalBias !== 0 && (e.normalBias = this.normalBias), this.radius !== 1 && (e.radius = this.radius), (this.mapSize.x !== 512 || this.mapSize.y !== 512) && (e.mapSize = this.mapSize.toArray()), e.camera = this.camera.toJSON(!1).object, delete e.camera.matrix, e;
  }
}
const ih = /* @__PURE__ */ new pt(), tr = /* @__PURE__ */ new N(), Ya = /* @__PURE__ */ new N();
class h0 extends bd {
  /**
   * Constructs a new point light shadow.
   */
  constructor() {
    super(new nn(90, 1, 0.5, 500)), this.isPointLightShadow = !0, this._frameExtents = new Ve(4, 2), this._viewportCount = 6, this._viewports = [
      // These viewports map a cube-map onto a 2D texture with the
      // following orientation:
      //
      //  xzXZ
      //   y Y
      //
      // X - Positive x direction
      // x - Negative x direction
      // Y - Positive y direction
      // y - Negative y direction
      // Z - Positive z direction
      // z - Negative z direction
      // positive X
      new lt(2, 1, 1, 1),
      // negative X
      new lt(0, 1, 1, 1),
      // positive Z
      new lt(3, 1, 1, 1),
      // negative Z
      new lt(1, 1, 1, 1),
      // positive Y
      new lt(3, 0, 1, 1),
      // negative Y
      new lt(1, 0, 1, 1)
    ], this._cubeDirections = [
      new N(1, 0, 0),
      new N(-1, 0, 0),
      new N(0, 0, 1),
      new N(0, 0, -1),
      new N(0, 1, 0),
      new N(0, -1, 0)
    ], this._cubeUps = [
      new N(0, 1, 0),
      new N(0, 1, 0),
      new N(0, 1, 0),
      new N(0, 1, 0),
      new N(0, 0, 1),
      new N(0, 0, -1)
    ];
  }
  /**
   * Update the matrices for the camera and shadow, used internally by the renderer.
   *
   * @param {Light} light - The light for which the shadow is being rendered.
   * @param {number} [viewportIndex=0] - The viewport index.
   */
  updateMatrices(e, t = 0) {
    const i = this.camera, s = this.matrix, r = e.distance || i.far;
    r !== i.far && (i.far = r, i.updateProjectionMatrix()), tr.setFromMatrixPosition(e.matrixWorld), i.position.copy(tr), Ya.copy(i.position), Ya.add(this._cubeDirections[t]), i.up.copy(this._cubeUps[t]), i.lookAt(Ya), i.updateMatrixWorld(), s.makeTranslation(-tr.x, -tr.y, -tr.z), ih.multiplyMatrices(i.projectionMatrix, i.matrixWorldInverse), this._frustum.setFromProjectionMatrix(ih, i.coordinateSystem, i.reversedDepth);
  }
}
class f0 extends Dc {
  /**
   * Constructs a new point light.
   *
   * @param {(number|Color|string)} [color=0xffffff] - The light's color.
   * @param {number} [intensity=1] - The light's strength/intensity measured in candela (cd).
   * @param {number} [distance=0] - Maximum range of the light. `0` means no limit.
   * @param {number} [decay=2] - The amount the light dims along the distance of the light.
   */
  constructor(e, t, i = 0, s = 2) {
    super(e, t), this.isPointLight = !0, this.type = "PointLight", this.distance = i, this.decay = s, this.shadow = new h0();
  }
  /**
   * The light's power. Power is the luminous power of the light measured in lumens (lm).
   * Changing the power will also change the light's intensity.
   *
   * @type {number}
   */
  get power() {
    return this.intensity * 4 * Math.PI;
  }
  set power(e) {
    this.intensity = e / (4 * Math.PI);
  }
  dispose() {
    this.shadow.dispose();
  }
  copy(e, t) {
    return super.copy(e, t), this.distance = e.distance, this.decay = e.decay, this.shadow = e.shadow.clone(), this;
  }
}
class Ad extends xd {
  /**
   * Constructs a new orthographic camera.
   *
   * @param {number} [left=-1] - The left plane of the camera's frustum.
   * @param {number} [right=1] - The right plane of the camera's frustum.
   * @param {number} [top=1] - The top plane of the camera's frustum.
   * @param {number} [bottom=-1] - The bottom plane of the camera's frustum.
   * @param {number} [near=0.1] - The camera's near plane.
   * @param {number} [far=2000] - The camera's far plane.
   */
  constructor(e = -1, t = 1, i = 1, s = -1, r = 0.1, o = 2e3) {
    super(), this.isOrthographicCamera = !0, this.type = "OrthographicCamera", this.zoom = 1, this.view = null, this.left = e, this.right = t, this.top = i, this.bottom = s, this.near = r, this.far = o, this.updateProjectionMatrix();
  }
  copy(e, t) {
    return super.copy(e, t), this.left = e.left, this.right = e.right, this.top = e.top, this.bottom = e.bottom, this.near = e.near, this.far = e.far, this.zoom = e.zoom, this.view = e.view === null ? null : Object.assign({}, e.view), this;
  }
  /**
   * Sets an offset in a larger frustum. This is useful for multi-window or
   * multi-monitor/multi-machine setups.
   *
   * @param {number} fullWidth - The full width of multiview setup.
   * @param {number} fullHeight - The full height of multiview setup.
   * @param {number} x - The horizontal offset of the subcamera.
   * @param {number} y - The vertical offset of the subcamera.
   * @param {number} width - The width of subcamera.
   * @param {number} height - The height of subcamera.
   * @see {@link PerspectiveCamera#setViewOffset}
   */
  setViewOffset(e, t, i, s, r, o) {
    this.view === null && (this.view = {
      enabled: !0,
      fullWidth: 1,
      fullHeight: 1,
      offsetX: 0,
      offsetY: 0,
      width: 1,
      height: 1
    }), this.view.enabled = !0, this.view.fullWidth = e, this.view.fullHeight = t, this.view.offsetX = i, this.view.offsetY = s, this.view.width = r, this.view.height = o, this.updateProjectionMatrix();
  }
  /**
   * Removes the view offset from the projection matrix.
   */
  clearViewOffset() {
    this.view !== null && (this.view.enabled = !1), this.updateProjectionMatrix();
  }
  /**
   * Updates the camera's projection matrix. Must be called after any change of
   * camera properties.
   */
  updateProjectionMatrix() {
    const e = (this.right - this.left) / (2 * this.zoom), t = (this.top - this.bottom) / (2 * this.zoom), i = (this.right + this.left) / 2, s = (this.top + this.bottom) / 2;
    let r = i - e, o = i + e, a = s + t, l = s - t;
    if (this.view !== null && this.view.enabled) {
      const c = (this.right - this.left) / this.view.fullWidth / this.zoom, u = (this.top - this.bottom) / this.view.fullHeight / this.zoom;
      r += c * this.view.offsetX, o = r + c * this.view.width, a -= u * this.view.offsetY, l = a - u * this.view.height;
    }
    this.projectionMatrix.makeOrthographic(r, o, a, l, this.near, this.far, this.coordinateSystem, this.reversedDepth), this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
  toJSON(e) {
    const t = super.toJSON(e);
    return t.object.zoom = this.zoom, t.object.left = this.left, t.object.right = this.right, t.object.top = this.top, t.object.bottom = this.bottom, t.object.near = this.near, t.object.far = this.far, this.view !== null && (t.object.view = Object.assign({}, this.view)), t;
  }
}
class d0 extends bd {
  /**
   * Constructs a new directional light shadow.
   */
  constructor() {
    super(new Ad(-5, 5, 5, -5, 0.5, 500)), this.isDirectionalLightShadow = !0;
  }
}
class p0 extends Dc {
  /**
   * Constructs a new directional light.
   *
   * @param {(number|Color|string)} [color=0xffffff] - The light's color.
   * @param {number} [intensity=1] - The light's strength/intensity.
   */
  constructor(e, t) {
    super(e, t), this.isDirectionalLight = !0, this.type = "DirectionalLight", this.position.copy(Tt.DEFAULT_UP), this.updateMatrix(), this.target = new Tt(), this.shadow = new d0();
  }
  dispose() {
    this.shadow.dispose();
  }
  copy(e) {
    return super.copy(e), this.target = e.target.clone(), this.shadow = e.shadow.clone(), this;
  }
}
class m0 extends nn {
  /**
   * Constructs a new array camera.
   *
   * @param {Array<PerspectiveCamera>} [array=[]] - An array of perspective sub cameras.
   */
  constructor(e = []) {
    super(), this.isArrayCamera = !0, this.isMultiViewCamera = !1, this.cameras = e;
  }
}
class _0 {
  /**
   * Constructs a new clock.
   *
   * @param {boolean} [autoStart=true] - Whether to automatically start the clock when
   * `getDelta()` is called for the first time.
   */
  constructor(e = !0) {
    this.autoStart = e, this.startTime = 0, this.oldTime = 0, this.elapsedTime = 0, this.running = !1;
  }
  /**
   * Starts the clock. When `autoStart` is set to `true`, the method is automatically
   * called by the class.
   */
  start() {
    this.startTime = performance.now(), this.oldTime = this.startTime, this.elapsedTime = 0, this.running = !0;
  }
  /**
   * Stops the clock.
   */
  stop() {
    this.getElapsedTime(), this.running = !1, this.autoStart = !1;
  }
  /**
   * Returns the elapsed time in seconds.
   *
   * @return {number} The elapsed time.
   */
  getElapsedTime() {
    return this.getDelta(), this.elapsedTime;
  }
  /**
   * Returns the delta time in seconds.
   *
   * @return {number} The delta time.
   */
  getDelta() {
    let e = 0;
    if (this.autoStart && !this.running)
      return this.start(), 0;
    if (this.running) {
      const t = performance.now();
      e = (t - this.oldTime) / 1e3, this.oldTime = t, this.elapsedTime += e;
    }
    return e;
  }
}
class sh {
  /**
   * Constructs a new spherical.
   *
   * @param {number} [radius=1] - The radius, or the Euclidean distance (straight-line distance) from the point to the origin.
   * @param {number} [phi=0] - The polar angle in radians from the y (up) axis.
   * @param {number} [theta=0] - The equator/azimuthal angle in radians around the y (up) axis.
   */
  constructor(e = 1, t = 0, i = 0) {
    this.radius = e, this.phi = t, this.theta = i;
  }
  /**
   * Sets the spherical components by copying the given values.
   *
   * @param {number} radius - The radius.
   * @param {number} phi - The polar angle.
   * @param {number} theta - The azimuthal angle.
   * @return {Spherical} A reference to this spherical.
   */
  set(e, t, i) {
    return this.radius = e, this.phi = t, this.theta = i, this;
  }
  /**
   * Copies the values of the given spherical to this instance.
   *
   * @param {Spherical} other - The spherical to copy.
   * @return {Spherical} A reference to this spherical.
   */
  copy(e) {
    return this.radius = e.radius, this.phi = e.phi, this.theta = e.theta, this;
  }
  /**
   * Restricts the polar angle [page:.phi phi] to be between `0.000001` and pi -
   * `0.000001`.
   *
   * @return {Spherical} A reference to this spherical.
   */
  makeSafe() {
    return this.phi = je(this.phi, 1e-6, Math.PI - 1e-6), this;
  }
  /**
   * Sets the spherical components from the given vector which is assumed to hold
   * Cartesian coordinates.
   *
   * @param {Vector3} v - The vector to set.
   * @return {Spherical} A reference to this spherical.
   */
  setFromVector3(e) {
    return this.setFromCartesianCoords(e.x, e.y, e.z);
  }
  /**
   * Sets the spherical components from the given Cartesian coordinates.
   *
   * @param {number} x - The x value.
   * @param {number} y - The y value.
   * @param {number} z - The z value.
   * @return {Spherical} A reference to this spherical.
   */
  setFromCartesianCoords(e, t, i) {
    return this.radius = Math.sqrt(e * e + t * t + i * i), this.radius === 0 ? (this.theta = 0, this.phi = 0) : (this.theta = Math.atan2(e, i), this.phi = Math.acos(je(t / this.radius, -1, 1))), this;
  }
  /**
   * Returns a new spherical with copied values from this instance.
   *
   * @return {Spherical} A clone of this instance.
   */
  clone() {
    return new this.constructor().copy(this);
  }
}
class rh extends Sd {
  /**
   * Constructs a new grid helper.
   *
   * @param {number} [size=10] - The size of the grid.
   * @param {number} [divisions=10] - The number of divisions across the grid.
   * @param {number|Color|string} [color1=0x444444] - The color of the center line.
   * @param {number|Color|string} [color2=0x888888] - The color of the lines of the grid.
   */
  constructor(e = 10, t = 10, i = 4473924, s = 8947848) {
    i = new We(i), s = new We(s);
    const r = t / 2, o = e / t, a = e / 2, l = [], c = [];
    for (let f = 0, p = 0, v = -a; f <= t; f++, v += o) {
      l.push(-a, 0, v, a, 0, v), l.push(v, 0, -a, v, 0, a);
      const x = f === r ? i : s;
      x.toArray(c, p), p += 3, x.toArray(c, p), p += 3, x.toArray(c, p), p += 3, x.toArray(c, p), p += 3;
    }
    const u = new Nt();
    u.setAttribute("position", new mt(l, 3)), u.setAttribute("color", new mt(c, 3));
    const h = new wc({ vertexColors: !0, toneMapped: !1 });
    super(u, h), this.type = "GridHelper";
  }
  /**
   * Frees the GPU-related resources allocated by this instance. Call this
   * method whenever this instance is no longer used in your app.
   */
  dispose() {
    this.geometry.dispose(), this.material.dispose();
  }
}
class g0 extends Zi {
  /**
   * Constructs a new controls instance.
   *
   * @param {Object3D} object - The object that is managed by the controls.
   * @param {?HTMLDOMElement} domElement - The HTML element used for event listeners.
   */
  constructor(e, t = null) {
    super(), this.object = e, this.domElement = t, this.enabled = !0, this.state = -1, this.keys = {}, this.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: null }, this.touches = { ONE: null, TWO: null };
  }
  /**
   * Connects the controls to the DOM. This method has so called "side effects" since
   * it adds the module's event listeners to the DOM.
   *
   * @param {HTMLDOMElement} element - The DOM element to connect to.
   */
  connect(e) {
    if (e === void 0) {
      console.warn("THREE.Controls: connect() now requires an element.");
      return;
    }
    this.domElement !== null && this.disconnect(), this.domElement = e;
  }
  /**
   * Disconnects the controls from the DOM.
   */
  disconnect() {
  }
  /**
   * Call this method if you no longer want use to the controls. It frees all internal
   * resources and removes all event listeners.
   */
  dispose() {
  }
  /**
   * Controls should implement this method if they have to update their internal state
   * per simulation step.
   *
   * @param {number} [delta] - The time delta in seconds.
   */
  update() {
  }
}
function oh(n, e, t, i) {
  const s = v0(i);
  switch (t) {
    // https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexImage2D.xhtml
    case od:
      return n * e;
    case ld:
      return n * e / s.components * s.byteLength;
    case yc:
      return n * e / s.components * s.byteLength;
    case cd:
      return n * e * 2 / s.components * s.byteLength;
    case Ec:
      return n * e * 2 / s.components * s.byteLength;
    case ad:
      return n * e * 3 / s.components * s.byteLength;
    case xn:
      return n * e * 4 / s.components * s.byteLength;
    case Tc:
      return n * e * 4 / s.components * s.byteLength;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_s3tc_srgb/
    case So:
    case yo:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 8;
    case Eo:
    case To:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_pvrtc/
    case Al:
    case Rl:
      return Math.max(n, 16) * Math.max(e, 8) / 4;
    case bl:
    case wl:
      return Math.max(n, 8) * Math.max(e, 8) / 2;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_etc/
    case Cl:
    case Pl:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 8;
    case Dl:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_astc/
    case Ll:
      return Math.floor((n + 3) / 4) * Math.floor((e + 3) / 4) * 16;
    case Il:
      return Math.floor((n + 4) / 5) * Math.floor((e + 3) / 4) * 16;
    case Ul:
      return Math.floor((n + 4) / 5) * Math.floor((e + 4) / 5) * 16;
    case Nl:
      return Math.floor((n + 5) / 6) * Math.floor((e + 4) / 5) * 16;
    case Fl:
      return Math.floor((n + 5) / 6) * Math.floor((e + 5) / 6) * 16;
    case Ol:
      return Math.floor((n + 7) / 8) * Math.floor((e + 4) / 5) * 16;
    case Bl:
      return Math.floor((n + 7) / 8) * Math.floor((e + 5) / 6) * 16;
    case zl:
      return Math.floor((n + 7) / 8) * Math.floor((e + 7) / 8) * 16;
    case Hl:
      return Math.floor((n + 9) / 10) * Math.floor((e + 4) / 5) * 16;
    case Vl:
      return Math.floor((n + 9) / 10) * Math.floor((e + 5) / 6) * 16;
    case kl:
      return Math.floor((n + 9) / 10) * Math.floor((e + 7) / 8) * 16;
    case Gl:
      return Math.floor((n + 9) / 10) * Math.floor((e + 9) / 10) * 16;
    case Wl:
      return Math.floor((n + 11) / 12) * Math.floor((e + 9) / 10) * 16;
    case Xl:
      return Math.floor((n + 11) / 12) * Math.floor((e + 11) / 12) * 16;
    // https://registry.khronos.org/webgl/extensions/EXT_texture_compression_bptc/
    case Yl:
    case ql:
    case jl:
      return Math.ceil(n / 4) * Math.ceil(e / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/EXT_texture_compression_rgtc/
    case Kl:
    case $l:
      return Math.ceil(n / 4) * Math.ceil(e / 4) * 8;
    case Zl:
    case Jl:
      return Math.ceil(n / 4) * Math.ceil(e / 4) * 16;
  }
  throw new Error(
    `Unable to determine texture byte length for ${t} format.`
  );
}
function v0(n) {
  switch (n) {
    case Bn:
    case nd:
      return { byteLength: 1, components: 1 };
    case Tr:
    case id:
    case Lr:
      return { byteLength: 2, components: 1 };
    case Mc:
    case Sc:
      return { byteLength: 2, components: 4 };
    case Xi:
    case xc:
    case ei:
      return { byteLength: 4, components: 1 };
    case sd:
    case rd:
      return { byteLength: 4, components: 3 };
  }
  throw new Error(`Unknown texture type ${n}.`);
}
typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register", { detail: {
  revision: vc
} }));
typeof window < "u" && (window.__THREE__ ? console.warn("WARNING: Multiple instances of Three.js being imported.") : window.__THREE__ = vc);
function wd() {
  let n = null, e = !1, t = null, i = null;
  function s(r, o) {
    t(r, o), i = n.requestAnimationFrame(s);
  }
  return {
    start: function() {
      e !== !0 && t !== null && (i = n.requestAnimationFrame(s), e = !0);
    },
    stop: function() {
      n.cancelAnimationFrame(i), e = !1;
    },
    setAnimationLoop: function(r) {
      t = r;
    },
    setContext: function(r) {
      n = r;
    }
  };
}
function x0(n) {
  const e = /* @__PURE__ */ new WeakMap();
  function t(a, l) {
    const c = a.array, u = a.usage, h = c.byteLength, f = n.createBuffer();
    n.bindBuffer(l, f), n.bufferData(l, c, u), a.onUploadCallback();
    let p;
    if (c instanceof Float32Array)
      p = n.FLOAT;
    else if (typeof Float16Array < "u" && c instanceof Float16Array)
      p = n.HALF_FLOAT;
    else if (c instanceof Uint16Array)
      a.isFloat16BufferAttribute ? p = n.HALF_FLOAT : p = n.UNSIGNED_SHORT;
    else if (c instanceof Int16Array)
      p = n.SHORT;
    else if (c instanceof Uint32Array)
      p = n.UNSIGNED_INT;
    else if (c instanceof Int32Array)
      p = n.INT;
    else if (c instanceof Int8Array)
      p = n.BYTE;
    else if (c instanceof Uint8Array)
      p = n.UNSIGNED_BYTE;
    else if (c instanceof Uint8ClampedArray)
      p = n.UNSIGNED_BYTE;
    else
      throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: " + c);
    return {
      buffer: f,
      type: p,
      bytesPerElement: c.BYTES_PER_ELEMENT,
      version: a.version,
      size: h
    };
  }
  function i(a, l, c) {
    const u = l.array, h = l.updateRanges;
    if (n.bindBuffer(c, a), h.length === 0)
      n.bufferSubData(c, 0, u);
    else {
      h.sort((p, v) => p.start - v.start);
      let f = 0;
      for (let p = 1; p < h.length; p++) {
        const v = h[f], x = h[p];
        x.start <= v.start + v.count + 1 ? v.count = Math.max(
          v.count,
          x.start + x.count - v.start
        ) : (++f, h[f] = x);
      }
      h.length = f + 1;
      for (let p = 0, v = h.length; p < v; p++) {
        const x = h[p];
        n.bufferSubData(
          c,
          x.start * u.BYTES_PER_ELEMENT,
          u,
          x.start,
          x.count
        );
      }
      l.clearUpdateRanges();
    }
    l.onUploadCallback();
  }
  function s(a) {
    return a.isInterleavedBufferAttribute && (a = a.data), e.get(a);
  }
  function r(a) {
    a.isInterleavedBufferAttribute && (a = a.data);
    const l = e.get(a);
    l && (n.deleteBuffer(l.buffer), e.delete(a));
  }
  function o(a, l) {
    if (a.isInterleavedBufferAttribute && (a = a.data), a.isGLBufferAttribute) {
      const u = e.get(a);
      (!u || u.version < a.version) && e.set(a, {
        buffer: a.buffer,
        type: a.type,
        bytesPerElement: a.elementSize,
        version: a.version
      });
      return;
    }
    const c = e.get(a);
    if (c === void 0)
      e.set(a, t(a, l));
    else if (c.version < a.version) {
      if (c.size !== a.array.byteLength)
        throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");
      i(c.buffer, a, l), c.version = a.version;
    }
  }
  return {
    get: s,
    remove: r,
    update: o
  };
}
var M0 = `#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`, S0 = `#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`, y0 = `#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`, E0 = `#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`, T0 = `#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`, b0 = `#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`, A0 = `#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT )
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN )
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`, w0 = `#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`, R0 = `#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`, C0 = `#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`, P0 = `vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`, D0 = `vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`, L0 = `float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`, I0 = `#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`, U0 = `#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`, N0 = `#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`, F0 = `#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`, O0 = `#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`, B0 = `#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`, z0 = `#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`, H0 = `#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`, V0 = `#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`, k0 = `#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`, G0 = `#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`, W0 = `#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`, X0 = `vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`, Y0 = `#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`, q0 = `#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`, j0 = `#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`, K0 = `#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`, $0 = "gl_FragColor = linearToOutputTexel( gl_FragColor );", Z0 = `vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`, J0 = `#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`, Q0 = `#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif

#endif`, ev = `#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`, tv = `#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS

		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`, nv = `#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`, iv = `#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`, sv = `#ifdef USE_FOG
	varying float vFogDepth;
#endif`, rv = `#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`, ov = `#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`, av = `#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`, lv = `#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`, cv = `LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`, uv = `varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`, hv = `uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`, fv = `#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`, dv = `ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`, pv = `varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`, mv = `BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`, _v = `varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`, gv = `PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`, vv = `struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`, xv = `
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`, Mv = `#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`, Sv = `#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`, yv = `#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`, Ev = `#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`, Tv = `#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`, bv = `#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`, Av = `#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`, wv = `#ifdef USE_MAP
	uniform sampler2D map;
#endif`, Rv = `#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`, Cv = `#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`, Pv = `float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`, Dv = `#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`, Lv = `#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`, Iv = `#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`, Uv = `#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`, Nv = `#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`, Fv = `#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`, Ov = `float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`, Bv = `#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`, zv = `#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`, Hv = `#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`, Vv = `#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`, kv = `#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`, Gv = `#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`, Wv = `#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`, Xv = `#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`, Yv = `#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`, qv = `#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`, jv = `vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`, Kv = `#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`, $v = `vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`, Zv = `#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`, Jv = `#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`, Qv = `float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`, ex = `#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`, tx = `#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		float depth = unpackRGBAToDepth( texture2D( depths, uv ) );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			return step( depth, compare );
		#else
			return step( compare, depth );
		#endif
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow( sampler2D shadow, vec2 uv, float compare ) {
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			float hard_shadow = step( distribution.x, compare );
		#else
			float hard_shadow = step( compare, distribution.x );
		#endif
		if ( hard_shadow != 1.0 ) {
			float distance = compare - distribution.x;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;

		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`, nx = `#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`, ix = `#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`, sx = `float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`, rx = `#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`, ox = `#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`, ax = `#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`, lx = `#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`, cx = `float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`, ux = `#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`, hx = `#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`, fx = `#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`, dx = `#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`, px = `#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`, mx = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`, _x = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`, gx = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`, vx = `#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;
const xx = `varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`, Mx = `uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, Sx = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`, yx = `#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, Ex = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`, Tx = `uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, bx = `#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`, Ax = `#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`, wx = `#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`, Rx = `#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`, Cx = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`, Px = `uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, Dx = `uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`, Lx = `uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`, Ix = `#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`, Ux = `uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Nx = `#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, Fx = `#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Ox = `#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`, Bx = `#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, zx = `#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`, Hx = `#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`, Vx = `#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, kx = `#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Gx = `#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`, Wx = `#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Xx = `#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, Yx = `#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, qx = `uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`, jx = `uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`, Kx = `#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, $x = `uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`, Zx = `uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`, Jx = `uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`, qe = {
  alphahash_fragment: M0,
  alphahash_pars_fragment: S0,
  alphamap_fragment: y0,
  alphamap_pars_fragment: E0,
  alphatest_fragment: T0,
  alphatest_pars_fragment: b0,
  aomap_fragment: A0,
  aomap_pars_fragment: w0,
  batching_pars_vertex: R0,
  batching_vertex: C0,
  begin_vertex: P0,
  beginnormal_vertex: D0,
  bsdfs: L0,
  iridescence_fragment: I0,
  bumpmap_pars_fragment: U0,
  clipping_planes_fragment: N0,
  clipping_planes_pars_fragment: F0,
  clipping_planes_pars_vertex: O0,
  clipping_planes_vertex: B0,
  color_fragment: z0,
  color_pars_fragment: H0,
  color_pars_vertex: V0,
  color_vertex: k0,
  common: G0,
  cube_uv_reflection_fragment: W0,
  defaultnormal_vertex: X0,
  displacementmap_pars_vertex: Y0,
  displacementmap_vertex: q0,
  emissivemap_fragment: j0,
  emissivemap_pars_fragment: K0,
  colorspace_fragment: $0,
  colorspace_pars_fragment: Z0,
  envmap_fragment: J0,
  envmap_common_pars_fragment: Q0,
  envmap_pars_fragment: ev,
  envmap_pars_vertex: tv,
  envmap_physical_pars_fragment: fv,
  envmap_vertex: nv,
  fog_vertex: iv,
  fog_pars_vertex: sv,
  fog_fragment: rv,
  fog_pars_fragment: ov,
  gradientmap_pars_fragment: av,
  lightmap_pars_fragment: lv,
  lights_lambert_fragment: cv,
  lights_lambert_pars_fragment: uv,
  lights_pars_begin: hv,
  lights_toon_fragment: dv,
  lights_toon_pars_fragment: pv,
  lights_phong_fragment: mv,
  lights_phong_pars_fragment: _v,
  lights_physical_fragment: gv,
  lights_physical_pars_fragment: vv,
  lights_fragment_begin: xv,
  lights_fragment_maps: Mv,
  lights_fragment_end: Sv,
  logdepthbuf_fragment: yv,
  logdepthbuf_pars_fragment: Ev,
  logdepthbuf_pars_vertex: Tv,
  logdepthbuf_vertex: bv,
  map_fragment: Av,
  map_pars_fragment: wv,
  map_particle_fragment: Rv,
  map_particle_pars_fragment: Cv,
  metalnessmap_fragment: Pv,
  metalnessmap_pars_fragment: Dv,
  morphinstance_vertex: Lv,
  morphcolor_vertex: Iv,
  morphnormal_vertex: Uv,
  morphtarget_pars_vertex: Nv,
  morphtarget_vertex: Fv,
  normal_fragment_begin: Ov,
  normal_fragment_maps: Bv,
  normal_pars_fragment: zv,
  normal_pars_vertex: Hv,
  normal_vertex: Vv,
  normalmap_pars_fragment: kv,
  clearcoat_normal_fragment_begin: Gv,
  clearcoat_normal_fragment_maps: Wv,
  clearcoat_pars_fragment: Xv,
  iridescence_pars_fragment: Yv,
  opaque_fragment: qv,
  packing: jv,
  premultiplied_alpha_fragment: Kv,
  project_vertex: $v,
  dithering_fragment: Zv,
  dithering_pars_fragment: Jv,
  roughnessmap_fragment: Qv,
  roughnessmap_pars_fragment: ex,
  shadowmap_pars_fragment: tx,
  shadowmap_pars_vertex: nx,
  shadowmap_vertex: ix,
  shadowmask_pars_fragment: sx,
  skinbase_vertex: rx,
  skinning_pars_vertex: ox,
  skinning_vertex: ax,
  skinnormal_vertex: lx,
  specularmap_fragment: cx,
  specularmap_pars_fragment: ux,
  tonemapping_fragment: hx,
  tonemapping_pars_fragment: fx,
  transmission_fragment: dx,
  transmission_pars_fragment: px,
  uv_pars_fragment: mx,
  uv_pars_vertex: _x,
  uv_vertex: gx,
  worldpos_vertex: vx,
  background_vert: xx,
  background_frag: Mx,
  backgroundCube_vert: Sx,
  backgroundCube_frag: yx,
  cube_vert: Ex,
  cube_frag: Tx,
  depth_vert: bx,
  depth_frag: Ax,
  distanceRGBA_vert: wx,
  distanceRGBA_frag: Rx,
  equirect_vert: Cx,
  equirect_frag: Px,
  linedashed_vert: Dx,
  linedashed_frag: Lx,
  meshbasic_vert: Ix,
  meshbasic_frag: Ux,
  meshlambert_vert: Nx,
  meshlambert_frag: Fx,
  meshmatcap_vert: Ox,
  meshmatcap_frag: Bx,
  meshnormal_vert: zx,
  meshnormal_frag: Hx,
  meshphong_vert: Vx,
  meshphong_frag: kx,
  meshphysical_vert: Gx,
  meshphysical_frag: Wx,
  meshtoon_vert: Xx,
  meshtoon_frag: Yx,
  points_vert: qx,
  points_frag: jx,
  shadow_vert: Kx,
  shadow_frag: $x,
  sprite_vert: Zx,
  sprite_frag: Jx
}, ve = {
  common: {
    diffuse: { value: /* @__PURE__ */ new We(16777215) },
    opacity: { value: 1 },
    map: { value: null },
    mapTransform: { value: /* @__PURE__ */ new Ye() },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new Ye() },
    alphaTest: { value: 0 }
  },
  specularmap: {
    specularMap: { value: null },
    specularMapTransform: { value: /* @__PURE__ */ new Ye() }
  },
  envmap: {
    envMap: { value: null },
    envMapRotation: { value: /* @__PURE__ */ new Ye() },
    flipEnvMap: { value: -1 },
    reflectivity: { value: 1 },
    // basic, lambert, phong
    ior: { value: 1.5 },
    // physical
    refractionRatio: { value: 0.98 }
    // basic, lambert, phong
  },
  aomap: {
    aoMap: { value: null },
    aoMapIntensity: { value: 1 },
    aoMapTransform: { value: /* @__PURE__ */ new Ye() }
  },
  lightmap: {
    lightMap: { value: null },
    lightMapIntensity: { value: 1 },
    lightMapTransform: { value: /* @__PURE__ */ new Ye() }
  },
  bumpmap: {
    bumpMap: { value: null },
    bumpMapTransform: { value: /* @__PURE__ */ new Ye() },
    bumpScale: { value: 1 }
  },
  normalmap: {
    normalMap: { value: null },
    normalMapTransform: { value: /* @__PURE__ */ new Ye() },
    normalScale: { value: /* @__PURE__ */ new Ve(1, 1) }
  },
  displacementmap: {
    displacementMap: { value: null },
    displacementMapTransform: { value: /* @__PURE__ */ new Ye() },
    displacementScale: { value: 1 },
    displacementBias: { value: 0 }
  },
  emissivemap: {
    emissiveMap: { value: null },
    emissiveMapTransform: { value: /* @__PURE__ */ new Ye() }
  },
  metalnessmap: {
    metalnessMap: { value: null },
    metalnessMapTransform: { value: /* @__PURE__ */ new Ye() }
  },
  roughnessmap: {
    roughnessMap: { value: null },
    roughnessMapTransform: { value: /* @__PURE__ */ new Ye() }
  },
  gradientmap: {
    gradientMap: { value: null }
  },
  fog: {
    fogDensity: { value: 25e-5 },
    fogNear: { value: 1 },
    fogFar: { value: 2e3 },
    fogColor: { value: /* @__PURE__ */ new We(16777215) }
  },
  lights: {
    ambientLightColor: { value: [] },
    lightProbe: { value: [] },
    directionalLights: { value: [], properties: {
      direction: {},
      color: {}
    } },
    directionalLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {}
    } },
    directionalShadowMap: { value: [] },
    directionalShadowMatrix: { value: [] },
    spotLights: { value: [], properties: {
      color: {},
      position: {},
      direction: {},
      distance: {},
      coneCos: {},
      penumbraCos: {},
      decay: {}
    } },
    spotLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {}
    } },
    spotLightMap: { value: [] },
    spotShadowMap: { value: [] },
    spotLightMatrix: { value: [] },
    pointLights: { value: [], properties: {
      color: {},
      position: {},
      decay: {},
      distance: {}
    } },
    pointLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {},
      shadowCameraNear: {},
      shadowCameraFar: {}
    } },
    pointShadowMap: { value: [] },
    pointShadowMatrix: { value: [] },
    hemisphereLights: { value: [], properties: {
      direction: {},
      skyColor: {},
      groundColor: {}
    } },
    // TODO (abelnation): RectAreaLight BRDF data needs to be moved from example to main src
    rectAreaLights: { value: [], properties: {
      color: {},
      position: {},
      width: {},
      height: {}
    } },
    ltc_1: { value: null },
    ltc_2: { value: null }
  },
  points: {
    diffuse: { value: /* @__PURE__ */ new We(16777215) },
    opacity: { value: 1 },
    size: { value: 1 },
    scale: { value: 1 },
    map: { value: null },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new Ye() },
    alphaTest: { value: 0 },
    uvTransform: { value: /* @__PURE__ */ new Ye() }
  },
  sprite: {
    diffuse: { value: /* @__PURE__ */ new We(16777215) },
    opacity: { value: 1 },
    center: { value: /* @__PURE__ */ new Ve(0.5, 0.5) },
    rotation: { value: 0 },
    map: { value: null },
    mapTransform: { value: /* @__PURE__ */ new Ye() },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new Ye() },
    alphaTest: { value: 0 }
  }
}, Ln = {
  basic: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.specularmap,
      ve.envmap,
      ve.aomap,
      ve.lightmap,
      ve.fog
    ]),
    vertexShader: qe.meshbasic_vert,
    fragmentShader: qe.meshbasic_frag
  },
  lambert: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.specularmap,
      ve.envmap,
      ve.aomap,
      ve.lightmap,
      ve.emissivemap,
      ve.bumpmap,
      ve.normalmap,
      ve.displacementmap,
      ve.fog,
      ve.lights,
      {
        emissive: { value: /* @__PURE__ */ new We(0) }
      }
    ]),
    vertexShader: qe.meshlambert_vert,
    fragmentShader: qe.meshlambert_frag
  },
  phong: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.specularmap,
      ve.envmap,
      ve.aomap,
      ve.lightmap,
      ve.emissivemap,
      ve.bumpmap,
      ve.normalmap,
      ve.displacementmap,
      ve.fog,
      ve.lights,
      {
        emissive: { value: /* @__PURE__ */ new We(0) },
        specular: { value: /* @__PURE__ */ new We(1118481) },
        shininess: { value: 30 }
      }
    ]),
    vertexShader: qe.meshphong_vert,
    fragmentShader: qe.meshphong_frag
  },
  standard: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.envmap,
      ve.aomap,
      ve.lightmap,
      ve.emissivemap,
      ve.bumpmap,
      ve.normalmap,
      ve.displacementmap,
      ve.roughnessmap,
      ve.metalnessmap,
      ve.fog,
      ve.lights,
      {
        emissive: { value: /* @__PURE__ */ new We(0) },
        roughness: { value: 1 },
        metalness: { value: 0 },
        envMapIntensity: { value: 1 }
      }
    ]),
    vertexShader: qe.meshphysical_vert,
    fragmentShader: qe.meshphysical_frag
  },
  toon: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.aomap,
      ve.lightmap,
      ve.emissivemap,
      ve.bumpmap,
      ve.normalmap,
      ve.displacementmap,
      ve.gradientmap,
      ve.fog,
      ve.lights,
      {
        emissive: { value: /* @__PURE__ */ new We(0) }
      }
    ]),
    vertexShader: qe.meshtoon_vert,
    fragmentShader: qe.meshtoon_frag
  },
  matcap: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.bumpmap,
      ve.normalmap,
      ve.displacementmap,
      ve.fog,
      {
        matcap: { value: null }
      }
    ]),
    vertexShader: qe.meshmatcap_vert,
    fragmentShader: qe.meshmatcap_frag
  },
  points: {
    uniforms: /* @__PURE__ */ zt([
      ve.points,
      ve.fog
    ]),
    vertexShader: qe.points_vert,
    fragmentShader: qe.points_frag
  },
  dashed: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.fog,
      {
        scale: { value: 1 },
        dashSize: { value: 1 },
        totalSize: { value: 2 }
      }
    ]),
    vertexShader: qe.linedashed_vert,
    fragmentShader: qe.linedashed_frag
  },
  depth: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.displacementmap
    ]),
    vertexShader: qe.depth_vert,
    fragmentShader: qe.depth_frag
  },
  normal: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.bumpmap,
      ve.normalmap,
      ve.displacementmap,
      {
        opacity: { value: 1 }
      }
    ]),
    vertexShader: qe.meshnormal_vert,
    fragmentShader: qe.meshnormal_frag
  },
  sprite: {
    uniforms: /* @__PURE__ */ zt([
      ve.sprite,
      ve.fog
    ]),
    vertexShader: qe.sprite_vert,
    fragmentShader: qe.sprite_frag
  },
  background: {
    uniforms: {
      uvTransform: { value: /* @__PURE__ */ new Ye() },
      t2D: { value: null },
      backgroundIntensity: { value: 1 }
    },
    vertexShader: qe.background_vert,
    fragmentShader: qe.background_frag
  },
  backgroundCube: {
    uniforms: {
      envMap: { value: null },
      flipEnvMap: { value: -1 },
      backgroundBlurriness: { value: 0 },
      backgroundIntensity: { value: 1 },
      backgroundRotation: { value: /* @__PURE__ */ new Ye() }
    },
    vertexShader: qe.backgroundCube_vert,
    fragmentShader: qe.backgroundCube_frag
  },
  cube: {
    uniforms: {
      tCube: { value: null },
      tFlip: { value: -1 },
      opacity: { value: 1 }
    },
    vertexShader: qe.cube_vert,
    fragmentShader: qe.cube_frag
  },
  equirect: {
    uniforms: {
      tEquirect: { value: null }
    },
    vertexShader: qe.equirect_vert,
    fragmentShader: qe.equirect_frag
  },
  distanceRGBA: {
    uniforms: /* @__PURE__ */ zt([
      ve.common,
      ve.displacementmap,
      {
        referencePosition: { value: /* @__PURE__ */ new N() },
        nearDistance: { value: 1 },
        farDistance: { value: 1e3 }
      }
    ]),
    vertexShader: qe.distanceRGBA_vert,
    fragmentShader: qe.distanceRGBA_frag
  },
  shadow: {
    uniforms: /* @__PURE__ */ zt([
      ve.lights,
      ve.fog,
      {
        color: { value: /* @__PURE__ */ new We(0) },
        opacity: { value: 1 }
      }
    ]),
    vertexShader: qe.shadow_vert,
    fragmentShader: qe.shadow_frag
  }
};
Ln.physical = {
  uniforms: /* @__PURE__ */ zt([
    Ln.standard.uniforms,
    {
      clearcoat: { value: 0 },
      clearcoatMap: { value: null },
      clearcoatMapTransform: { value: /* @__PURE__ */ new Ye() },
      clearcoatNormalMap: { value: null },
      clearcoatNormalMapTransform: { value: /* @__PURE__ */ new Ye() },
      clearcoatNormalScale: { value: /* @__PURE__ */ new Ve(1, 1) },
      clearcoatRoughness: { value: 0 },
      clearcoatRoughnessMap: { value: null },
      clearcoatRoughnessMapTransform: { value: /* @__PURE__ */ new Ye() },
      dispersion: { value: 0 },
      iridescence: { value: 0 },
      iridescenceMap: { value: null },
      iridescenceMapTransform: { value: /* @__PURE__ */ new Ye() },
      iridescenceIOR: { value: 1.3 },
      iridescenceThicknessMinimum: { value: 100 },
      iridescenceThicknessMaximum: { value: 400 },
      iridescenceThicknessMap: { value: null },
      iridescenceThicknessMapTransform: { value: /* @__PURE__ */ new Ye() },
      sheen: { value: 0 },
      sheenColor: { value: /* @__PURE__ */ new We(0) },
      sheenColorMap: { value: null },
      sheenColorMapTransform: { value: /* @__PURE__ */ new Ye() },
      sheenRoughness: { value: 1 },
      sheenRoughnessMap: { value: null },
      sheenRoughnessMapTransform: { value: /* @__PURE__ */ new Ye() },
      transmission: { value: 0 },
      transmissionMap: { value: null },
      transmissionMapTransform: { value: /* @__PURE__ */ new Ye() },
      transmissionSamplerSize: { value: /* @__PURE__ */ new Ve() },
      transmissionSamplerMap: { value: null },
      thickness: { value: 0 },
      thicknessMap: { value: null },
      thicknessMapTransform: { value: /* @__PURE__ */ new Ye() },
      attenuationDistance: { value: 0 },
      attenuationColor: { value: /* @__PURE__ */ new We(0) },
      specularColor: { value: /* @__PURE__ */ new We(1, 1, 1) },
      specularColorMap: { value: null },
      specularColorMapTransform: { value: /* @__PURE__ */ new Ye() },
      specularIntensity: { value: 1 },
      specularIntensityMap: { value: null },
      specularIntensityMapTransform: { value: /* @__PURE__ */ new Ye() },
      anisotropyVector: { value: /* @__PURE__ */ new Ve() },
      anisotropyMap: { value: null },
      anisotropyMapTransform: { value: /* @__PURE__ */ new Ye() }
    }
  ]),
  vertexShader: qe.meshphysical_vert,
  fragmentShader: qe.meshphysical_frag
};
const fo = { r: 0, b: 0, g: 0 }, Ii = /* @__PURE__ */ new zn(), Qx = /* @__PURE__ */ new pt();
function eM(n, e, t, i, s, r, o) {
  const a = new We(0);
  let l = r === !0 ? 0 : 1, c, u, h = null, f = 0, p = null;
  function v(A) {
    let M = A.isScene === !0 ? A.background : null;
    return M && M.isTexture && (M = (A.backgroundBlurriness > 0 ? t : e).get(M)), M;
  }
  function x(A) {
    let M = !1;
    const R = v(A);
    R === null ? d(a, l) : R && R.isColor && (d(R, 1), M = !0);
    const w = n.xr.getEnvironmentBlendMode();
    w === "additive" ? i.buffers.color.setClear(0, 0, 0, 1, o) : w === "alpha-blend" && i.buffers.color.setClear(0, 0, 0, 0, o), (n.autoClear || M) && (i.buffers.depth.setTest(!0), i.buffers.depth.setMask(!0), i.buffers.color.setMask(!0), n.clear(n.autoClearColor, n.autoClearDepth, n.autoClearStencil));
  }
  function m(A, M) {
    const R = v(M);
    R && (R.isCubeTexture || R.mapping === ta) ? (u === void 0 && (u = new vt(
      new ji(1, 1, 1),
      new yi({
        name: "BackgroundCubeMaterial",
        uniforms: Bs(Ln.backgroundCube.uniforms),
        vertexShader: Ln.backgroundCube.vertexShader,
        fragmentShader: Ln.backgroundCube.fragmentShader,
        side: Wt,
        depthTest: !1,
        depthWrite: !1,
        fog: !1,
        allowOverride: !1
      })
    ), u.geometry.deleteAttribute("normal"), u.geometry.deleteAttribute("uv"), u.onBeforeRender = function(w, D, U) {
      this.matrixWorld.copyPosition(U.matrixWorld);
    }, Object.defineProperty(u.material, "envMap", {
      get: function() {
        return this.uniforms.envMap.value;
      }
    }), s.update(u)), Ii.copy(M.backgroundRotation), Ii.x *= -1, Ii.y *= -1, Ii.z *= -1, R.isCubeTexture && R.isRenderTargetTexture === !1 && (Ii.y *= -1, Ii.z *= -1), u.material.uniforms.envMap.value = R, u.material.uniforms.flipEnvMap.value = R.isCubeTexture && R.isRenderTargetTexture === !1 ? -1 : 1, u.material.uniforms.backgroundBlurriness.value = M.backgroundBlurriness, u.material.uniforms.backgroundIntensity.value = M.backgroundIntensity, u.material.uniforms.backgroundRotation.value.setFromMatrix4(Qx.makeRotationFromEuler(Ii)), u.material.toneMapped = Qe.getTransfer(R.colorSpace) !== ot, (h !== R || f !== R.version || p !== n.toneMapping) && (u.material.needsUpdate = !0, h = R, f = R.version, p = n.toneMapping), u.layers.enableAll(), A.unshift(u, u.geometry, u.material, 0, 0, null)) : R && R.isTexture && (c === void 0 && (c = new vt(
      new zs(2, 2),
      new yi({
        name: "BackgroundMaterial",
        uniforms: Bs(Ln.background.uniforms),
        vertexShader: Ln.background.vertexShader,
        fragmentShader: Ln.background.fragmentShader,
        side: Si,
        depthTest: !1,
        depthWrite: !1,
        fog: !1,
        allowOverride: !1
      })
    ), c.geometry.deleteAttribute("normal"), Object.defineProperty(c.material, "map", {
      get: function() {
        return this.uniforms.t2D.value;
      }
    }), s.update(c)), c.material.uniforms.t2D.value = R, c.material.uniforms.backgroundIntensity.value = M.backgroundIntensity, c.material.toneMapped = Qe.getTransfer(R.colorSpace) !== ot, R.matrixAutoUpdate === !0 && R.updateMatrix(), c.material.uniforms.uvTransform.value.copy(R.matrix), (h !== R || f !== R.version || p !== n.toneMapping) && (c.material.needsUpdate = !0, h = R, f = R.version, p = n.toneMapping), c.layers.enableAll(), A.unshift(c, c.geometry, c.material, 0, 0, null));
  }
  function d(A, M) {
    A.getRGB(fo, vd(n)), i.buffers.color.setClear(fo.r, fo.g, fo.b, M, o);
  }
  function b() {
    u !== void 0 && (u.geometry.dispose(), u.material.dispose(), u = void 0), c !== void 0 && (c.geometry.dispose(), c.material.dispose(), c = void 0);
  }
  return {
    getClearColor: function() {
      return a;
    },
    setClearColor: function(A, M = 1) {
      a.set(A), l = M, d(a, l);
    },
    getClearAlpha: function() {
      return l;
    },
    setClearAlpha: function(A) {
      l = A, d(a, l);
    },
    render: x,
    addToRenderList: m,
    dispose: b
  };
}
function tM(n, e) {
  const t = n.getParameter(n.MAX_VERTEX_ATTRIBS), i = {}, s = f(null);
  let r = s, o = !1;
  function a(S, P, L, V, Z) {
    let te = !1;
    const $ = h(V, L, P);
    r !== $ && (r = $, c(r.object)), te = p(S, V, L, Z), te && v(S, V, L, Z), Z !== null && e.update(Z, n.ELEMENT_ARRAY_BUFFER), (te || o) && (o = !1, M(S, P, L, V), Z !== null && n.bindBuffer(n.ELEMENT_ARRAY_BUFFER, e.get(Z).buffer));
  }
  function l() {
    return n.createVertexArray();
  }
  function c(S) {
    return n.bindVertexArray(S);
  }
  function u(S) {
    return n.deleteVertexArray(S);
  }
  function h(S, P, L) {
    const V = L.wireframe === !0;
    let Z = i[S.id];
    Z === void 0 && (Z = {}, i[S.id] = Z);
    let te = Z[P.id];
    te === void 0 && (te = {}, Z[P.id] = te);
    let $ = te[V];
    return $ === void 0 && ($ = f(l()), te[V] = $), $;
  }
  function f(S) {
    const P = [], L = [], V = [];
    for (let Z = 0; Z < t; Z++)
      P[Z] = 0, L[Z] = 0, V[Z] = 0;
    return {
      // for backward compatibility on non-VAO support browser
      geometry: null,
      program: null,
      wireframe: !1,
      newAttributes: P,
      enabledAttributes: L,
      attributeDivisors: V,
      object: S,
      attributes: {},
      index: null
    };
  }
  function p(S, P, L, V) {
    const Z = r.attributes, te = P.attributes;
    let $ = 0;
    const ie = L.getAttributes();
    for (const H in ie)
      if (ie[H].location >= 0) {
        const xe = Z[H];
        let me = te[H];
        if (me === void 0 && (H === "instanceMatrix" && S.instanceMatrix && (me = S.instanceMatrix), H === "instanceColor" && S.instanceColor && (me = S.instanceColor)), xe === void 0 || xe.attribute !== me || me && xe.data !== me.data) return !0;
        $++;
      }
    return r.attributesNum !== $ || r.index !== V;
  }
  function v(S, P, L, V) {
    const Z = {}, te = P.attributes;
    let $ = 0;
    const ie = L.getAttributes();
    for (const H in ie)
      if (ie[H].location >= 0) {
        let xe = te[H];
        xe === void 0 && (H === "instanceMatrix" && S.instanceMatrix && (xe = S.instanceMatrix), H === "instanceColor" && S.instanceColor && (xe = S.instanceColor));
        const me = {};
        me.attribute = xe, xe && xe.data && (me.data = xe.data), Z[H] = me, $++;
      }
    r.attributes = Z, r.attributesNum = $, r.index = V;
  }
  function x() {
    const S = r.newAttributes;
    for (let P = 0, L = S.length; P < L; P++)
      S[P] = 0;
  }
  function m(S) {
    d(S, 0);
  }
  function d(S, P) {
    const L = r.newAttributes, V = r.enabledAttributes, Z = r.attributeDivisors;
    L[S] = 1, V[S] === 0 && (n.enableVertexAttribArray(S), V[S] = 1), Z[S] !== P && (n.vertexAttribDivisor(S, P), Z[S] = P);
  }
  function b() {
    const S = r.newAttributes, P = r.enabledAttributes;
    for (let L = 0, V = P.length; L < V; L++)
      P[L] !== S[L] && (n.disableVertexAttribArray(L), P[L] = 0);
  }
  function A(S, P, L, V, Z, te, $) {
    $ === !0 ? n.vertexAttribIPointer(S, P, L, Z, te) : n.vertexAttribPointer(S, P, L, V, Z, te);
  }
  function M(S, P, L, V) {
    x();
    const Z = V.attributes, te = L.getAttributes(), $ = P.defaultAttributeValues;
    for (const ie in te) {
      const H = te[ie];
      if (H.location >= 0) {
        let fe = Z[ie];
        if (fe === void 0 && (ie === "instanceMatrix" && S.instanceMatrix && (fe = S.instanceMatrix), ie === "instanceColor" && S.instanceColor && (fe = S.instanceColor)), fe !== void 0) {
          const xe = fe.normalized, me = fe.itemSize, de = e.get(fe);
          if (de === void 0) continue;
          const Le = de.buffer, tt = de.type, Ze = de.bytesPerElement, ne = tt === n.INT || tt === n.UNSIGNED_INT || fe.gpuType === xc;
          if (fe.isInterleavedBufferAttribute) {
            const re = fe.data, Ae = re.stride, Oe = fe.offset;
            if (re.isInstancedInterleavedBuffer) {
              for (let Pe = 0; Pe < H.locationSize; Pe++)
                d(H.location + Pe, re.meshPerAttribute);
              S.isInstancedMesh !== !0 && V._maxInstanceCount === void 0 && (V._maxInstanceCount = re.meshPerAttribute * re.count);
            } else
              for (let Pe = 0; Pe < H.locationSize; Pe++)
                m(H.location + Pe);
            n.bindBuffer(n.ARRAY_BUFFER, Le);
            for (let Pe = 0; Pe < H.locationSize; Pe++)
              A(
                H.location + Pe,
                me / H.locationSize,
                tt,
                xe,
                Ae * Ze,
                (Oe + me / H.locationSize * Pe) * Ze,
                ne
              );
          } else {
            if (fe.isInstancedBufferAttribute) {
              for (let re = 0; re < H.locationSize; re++)
                d(H.location + re, fe.meshPerAttribute);
              S.isInstancedMesh !== !0 && V._maxInstanceCount === void 0 && (V._maxInstanceCount = fe.meshPerAttribute * fe.count);
            } else
              for (let re = 0; re < H.locationSize; re++)
                m(H.location + re);
            n.bindBuffer(n.ARRAY_BUFFER, Le);
            for (let re = 0; re < H.locationSize; re++)
              A(
                H.location + re,
                me / H.locationSize,
                tt,
                xe,
                me * Ze,
                me / H.locationSize * re * Ze,
                ne
              );
          }
        } else if ($ !== void 0) {
          const xe = $[ie];
          if (xe !== void 0)
            switch (xe.length) {
              case 2:
                n.vertexAttrib2fv(H.location, xe);
                break;
              case 3:
                n.vertexAttrib3fv(H.location, xe);
                break;
              case 4:
                n.vertexAttrib4fv(H.location, xe);
                break;
              default:
                n.vertexAttrib1fv(H.location, xe);
            }
        }
      }
    }
    b();
  }
  function R() {
    U();
    for (const S in i) {
      const P = i[S];
      for (const L in P) {
        const V = P[L];
        for (const Z in V)
          u(V[Z].object), delete V[Z];
        delete P[L];
      }
      delete i[S];
    }
  }
  function w(S) {
    if (i[S.id] === void 0) return;
    const P = i[S.id];
    for (const L in P) {
      const V = P[L];
      for (const Z in V)
        u(V[Z].object), delete V[Z];
      delete P[L];
    }
    delete i[S.id];
  }
  function D(S) {
    for (const P in i) {
      const L = i[P];
      if (L[S.id] === void 0) continue;
      const V = L[S.id];
      for (const Z in V)
        u(V[Z].object), delete V[Z];
      delete L[S.id];
    }
  }
  function U() {
    y(), o = !0, r !== s && (r = s, c(r.object));
  }
  function y() {
    s.geometry = null, s.program = null, s.wireframe = !1;
  }
  return {
    setup: a,
    reset: U,
    resetDefaultState: y,
    dispose: R,
    releaseStatesOfGeometry: w,
    releaseStatesOfProgram: D,
    initAttributes: x,
    enableAttribute: m,
    disableUnusedAttributes: b
  };
}
function nM(n, e, t) {
  let i;
  function s(c) {
    i = c;
  }
  function r(c, u) {
    n.drawArrays(i, c, u), t.update(u, i, 1);
  }
  function o(c, u, h) {
    h !== 0 && (n.drawArraysInstanced(i, c, u, h), t.update(u, i, h));
  }
  function a(c, u, h) {
    if (h === 0) return;
    e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i, c, 0, u, 0, h);
    let p = 0;
    for (let v = 0; v < h; v++)
      p += u[v];
    t.update(p, i, 1);
  }
  function l(c, u, h, f) {
    if (h === 0) return;
    const p = e.get("WEBGL_multi_draw");
    if (p === null)
      for (let v = 0; v < c.length; v++)
        o(c[v], u[v], f[v]);
    else {
      p.multiDrawArraysInstancedWEBGL(i, c, 0, u, 0, f, 0, h);
      let v = 0;
      for (let x = 0; x < h; x++)
        v += u[x] * f[x];
      t.update(v, i, 1);
    }
  }
  this.setMode = s, this.render = r, this.renderInstances = o, this.renderMultiDraw = a, this.renderMultiDrawInstances = l;
}
function iM(n, e, t, i) {
  let s;
  function r() {
    if (s !== void 0) return s;
    if (e.has("EXT_texture_filter_anisotropic") === !0) {
      const D = e.get("EXT_texture_filter_anisotropic");
      s = n.getParameter(D.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
    } else
      s = 0;
    return s;
  }
  function o(D) {
    return !(D !== xn && i.convert(D) !== n.getParameter(n.IMPLEMENTATION_COLOR_READ_FORMAT));
  }
  function a(D) {
    const U = D === Lr && (e.has("EXT_color_buffer_half_float") || e.has("EXT_color_buffer_float"));
    return !(D !== Bn && i.convert(D) !== n.getParameter(n.IMPLEMENTATION_COLOR_READ_TYPE) && // Edge and Chrome Mac < 52 (#9513)
    D !== ei && !U);
  }
  function l(D) {
    if (D === "highp") {
      if (n.getShaderPrecisionFormat(n.VERTEX_SHADER, n.HIGH_FLOAT).precision > 0 && n.getShaderPrecisionFormat(n.FRAGMENT_SHADER, n.HIGH_FLOAT).precision > 0)
        return "highp";
      D = "mediump";
    }
    return D === "mediump" && n.getShaderPrecisionFormat(n.VERTEX_SHADER, n.MEDIUM_FLOAT).precision > 0 && n.getShaderPrecisionFormat(n.FRAGMENT_SHADER, n.MEDIUM_FLOAT).precision > 0 ? "mediump" : "lowp";
  }
  let c = t.precision !== void 0 ? t.precision : "highp";
  const u = l(c);
  u !== c && (console.warn("THREE.WebGLRenderer:", c, "not supported, using", u, "instead."), c = u);
  const h = t.logarithmicDepthBuffer === !0, f = t.reversedDepthBuffer === !0 && e.has("EXT_clip_control"), p = n.getParameter(n.MAX_TEXTURE_IMAGE_UNITS), v = n.getParameter(n.MAX_VERTEX_TEXTURE_IMAGE_UNITS), x = n.getParameter(n.MAX_TEXTURE_SIZE), m = n.getParameter(n.MAX_CUBE_MAP_TEXTURE_SIZE), d = n.getParameter(n.MAX_VERTEX_ATTRIBS), b = n.getParameter(n.MAX_VERTEX_UNIFORM_VECTORS), A = n.getParameter(n.MAX_VARYING_VECTORS), M = n.getParameter(n.MAX_FRAGMENT_UNIFORM_VECTORS), R = v > 0, w = n.getParameter(n.MAX_SAMPLES);
  return {
    isWebGL2: !0,
    // keeping this for backwards compatibility
    getMaxAnisotropy: r,
    getMaxPrecision: l,
    textureFormatReadable: o,
    textureTypeReadable: a,
    precision: c,
    logarithmicDepthBuffer: h,
    reversedDepthBuffer: f,
    maxTextures: p,
    maxVertexTextures: v,
    maxTextureSize: x,
    maxCubemapSize: m,
    maxAttributes: d,
    maxVertexUniforms: b,
    maxVaryings: A,
    maxFragmentUniforms: M,
    vertexTextures: R,
    maxSamples: w
  };
}
function sM(n) {
  const e = this;
  let t = null, i = 0, s = !1, r = !1;
  const o = new mi(), a = new Ye(), l = { value: null, needsUpdate: !1 };
  this.uniform = l, this.numPlanes = 0, this.numIntersection = 0, this.init = function(h, f) {
    const p = h.length !== 0 || f || // enable state of previous frame - the clipping code has to
    // run another frame in order to reset the state:
    i !== 0 || s;
    return s = f, i = h.length, p;
  }, this.beginShadows = function() {
    r = !0, u(null);
  }, this.endShadows = function() {
    r = !1;
  }, this.setGlobalState = function(h, f) {
    t = u(h, f, 0);
  }, this.setState = function(h, f, p) {
    const v = h.clippingPlanes, x = h.clipIntersection, m = h.clipShadows, d = n.get(h);
    if (!s || v === null || v.length === 0 || r && !m)
      r ? u(null) : c();
    else {
      const b = r ? 0 : i, A = b * 4;
      let M = d.clippingState || null;
      l.value = M, M = u(v, f, A, p);
      for (let R = 0; R !== A; ++R)
        M[R] = t[R];
      d.clippingState = M, this.numIntersection = x ? this.numPlanes : 0, this.numPlanes += b;
    }
  };
  function c() {
    l.value !== t && (l.value = t, l.needsUpdate = i > 0), e.numPlanes = i, e.numIntersection = 0;
  }
  function u(h, f, p, v) {
    const x = h !== null ? h.length : 0;
    let m = null;
    if (x !== 0) {
      if (m = l.value, v !== !0 || m === null) {
        const d = p + x * 4, b = f.matrixWorldInverse;
        a.getNormalMatrix(b), (m === null || m.length < d) && (m = new Float32Array(d));
        for (let A = 0, M = p; A !== x; ++A, M += 4)
          o.copy(h[A]).applyMatrix4(b, a), o.normal.toArray(m, M), m[M + 3] = o.constant;
      }
      l.value = m, l.needsUpdate = !0;
    }
    return e.numPlanes = x, e.numIntersection = 0, m;
  }
}
function rM(n) {
  let e = /* @__PURE__ */ new WeakMap();
  function t(o, a) {
    return a === Sl ? o.mapping = Ns : a === yl && (o.mapping = Fs), o;
  }
  function i(o) {
    if (o && o.isTexture) {
      const a = o.mapping;
      if (a === Sl || a === yl)
        if (e.has(o)) {
          const l = e.get(o).texture;
          return t(l, o.mapping);
        } else {
          const l = o.image;
          if (l && l.height > 0) {
            const c = new Qg(l.height);
            return c.fromEquirectangularTexture(n, o), e.set(o, c), o.addEventListener("dispose", s), t(c.texture, o.mapping);
          } else
            return null;
        }
    }
    return o;
  }
  function s(o) {
    const a = o.target;
    a.removeEventListener("dispose", s);
    const l = e.get(a);
    l !== void 0 && (e.delete(a), l.dispose());
  }
  function r() {
    e = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: i,
    dispose: r
  };
}
const Ts = 4, ah = [0.125, 0.215, 0.35, 0.446, 0.526, 0.582], zi = 20, qa = /* @__PURE__ */ new Ad(), lh = /* @__PURE__ */ new We();
let ja = null, Ka = 0, $a = 0, Za = !1;
const Fi = (1 + Math.sqrt(5)) / 2, ms = 1 / Fi, ch = [
  /* @__PURE__ */ new N(-Fi, ms, 0),
  /* @__PURE__ */ new N(Fi, ms, 0),
  /* @__PURE__ */ new N(-ms, 0, Fi),
  /* @__PURE__ */ new N(ms, 0, Fi),
  /* @__PURE__ */ new N(0, Fi, -ms),
  /* @__PURE__ */ new N(0, Fi, ms),
  /* @__PURE__ */ new N(-1, 1, -1),
  /* @__PURE__ */ new N(1, 1, -1),
  /* @__PURE__ */ new N(-1, 1, 1),
  /* @__PURE__ */ new N(1, 1, 1)
], oM = /* @__PURE__ */ new N();
class uh {
  /**
   * Constructs a new PMREM generator.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   */
  constructor(e) {
    this._renderer = e, this._pingPongRenderTarget = null, this._lodMax = 0, this._cubeSize = 0, this._lodPlanes = [], this._sizeLods = [], this._sigmas = [], this._blurMaterial = null, this._cubemapMaterial = null, this._equirectMaterial = null, this._compileMaterial(this._blurMaterial);
  }
  /**
   * Generates a PMREM from a supplied Scene, which can be faster than using an
   * image if networking bandwidth is low. Optional sigma specifies a blur radius
   * in radians to be applied to the scene before PMREM generation. Optional near
   * and far planes ensure the scene is rendered in its entirety.
   *
   * @param {Scene} scene - The scene to be captured.
   * @param {number} [sigma=0] - The blur radius in radians.
   * @param {number} [near=0.1] - The near plane distance.
   * @param {number} [far=100] - The far plane distance.
   * @param {Object} [options={}] - The configuration options.
   * @param {number} [options.size=256] - The texture size of the PMREM.
   * @param {Vector3} [options.renderTarget=origin] - The position of the internal cube camera that renders the scene.
   * @return {WebGLRenderTarget} The resulting PMREM.
   */
  fromScene(e, t = 0, i = 0.1, s = 100, r = {}) {
    const {
      size: o = 256,
      position: a = oM
    } = r;
    ja = this._renderer.getRenderTarget(), Ka = this._renderer.getActiveCubeFace(), $a = this._renderer.getActiveMipmapLevel(), Za = this._renderer.xr.enabled, this._renderer.xr.enabled = !1, this._setSize(o);
    const l = this._allocateTargets();
    return l.depthBuffer = !0, this._sceneToCubeUV(e, i, s, l, a), t > 0 && this._blur(l, 0, 0, t), this._applyPMREM(l), this._cleanup(l), l;
  }
  /**
   * Generates a PMREM from an equirectangular texture, which can be either LDR
   * or HDR. The ideal input image size is 1k (1024 x 512),
   * as this matches best with the 256 x 256 cubemap output.
   *
   * @param {Texture} equirectangular - The equirectangular texture to be converted.
   * @param {?WebGLRenderTarget} [renderTarget=null] - The render target to use.
   * @return {WebGLRenderTarget} The resulting PMREM.
   */
  fromEquirectangular(e, t = null) {
    return this._fromTexture(e, t);
  }
  /**
   * Generates a PMREM from an cubemap texture, which can be either LDR
   * or HDR. The ideal input cube size is 256 x 256,
   * as this matches best with the 256 x 256 cubemap output.
   *
   * @param {Texture} cubemap - The cubemap texture to be converted.
   * @param {?WebGLRenderTarget} [renderTarget=null] - The render target to use.
   * @return {WebGLRenderTarget} The resulting PMREM.
   */
  fromCubemap(e, t = null) {
    return this._fromTexture(e, t);
  }
  /**
   * Pre-compiles the cubemap shader. You can get faster start-up by invoking this method during
   * your texture's network fetch for increased concurrency.
   */
  compileCubemapShader() {
    this._cubemapMaterial === null && (this._cubemapMaterial = dh(), this._compileMaterial(this._cubemapMaterial));
  }
  /**
   * Pre-compiles the equirectangular shader. You can get faster start-up by invoking this method during
   * your texture's network fetch for increased concurrency.
   */
  compileEquirectangularShader() {
    this._equirectMaterial === null && (this._equirectMaterial = fh(), this._compileMaterial(this._equirectMaterial));
  }
  /**
   * Disposes of the PMREMGenerator's internal memory. Note that PMREMGenerator is a static class,
   * so you should not need more than one PMREMGenerator object. If you do, calling dispose() on
   * one of them will cause any others to also become unusable.
   */
  dispose() {
    this._dispose(), this._cubemapMaterial !== null && this._cubemapMaterial.dispose(), this._equirectMaterial !== null && this._equirectMaterial.dispose();
  }
  // private interface
  _setSize(e) {
    this._lodMax = Math.floor(Math.log2(e)), this._cubeSize = Math.pow(2, this._lodMax);
  }
  _dispose() {
    this._blurMaterial !== null && this._blurMaterial.dispose(), this._pingPongRenderTarget !== null && this._pingPongRenderTarget.dispose();
    for (let e = 0; e < this._lodPlanes.length; e++)
      this._lodPlanes[e].dispose();
  }
  _cleanup(e) {
    this._renderer.setRenderTarget(ja, Ka, $a), this._renderer.xr.enabled = Za, e.scissorTest = !1, po(e, 0, 0, e.width, e.height);
  }
  _fromTexture(e, t) {
    e.mapping === Ns || e.mapping === Fs ? this._setSize(e.image.length === 0 ? 16 : e.image[0].width || e.image[0].image.width) : this._setSize(e.image.width / 4), ja = this._renderer.getRenderTarget(), Ka = this._renderer.getActiveCubeFace(), $a = this._renderer.getActiveMipmapLevel(), Za = this._renderer.xr.enabled, this._renderer.xr.enabled = !1;
    const i = t || this._allocateTargets();
    return this._textureToCubeUV(e, i), this._applyPMREM(i), this._cleanup(i), i;
  }
  _allocateTargets() {
    const e = 3 * Math.max(this._cubeSize, 112), t = 4 * this._cubeSize, i = {
      magFilter: Un,
      minFilter: Un,
      generateMipmaps: !1,
      type: Lr,
      format: xn,
      colorSpace: Os,
      depthBuffer: !1
    }, s = hh(e, t, i);
    if (this._pingPongRenderTarget === null || this._pingPongRenderTarget.width !== e || this._pingPongRenderTarget.height !== t) {
      this._pingPongRenderTarget !== null && this._dispose(), this._pingPongRenderTarget = hh(e, t, i);
      const { _lodMax: r } = this;
      ({ sizeLods: this._sizeLods, lodPlanes: this._lodPlanes, sigmas: this._sigmas } = aM(r)), this._blurMaterial = lM(r, e, t);
    }
    return s;
  }
  _compileMaterial(e) {
    const t = new vt(this._lodPlanes[0], e);
    this._renderer.compile(t, qa);
  }
  _sceneToCubeUV(e, t, i, s, r) {
    const l = new nn(90, 1, t, i), c = [1, -1, 1, 1, 1, 1], u = [1, 1, 1, -1, -1, -1], h = this._renderer, f = h.autoClear, p = h.toneMapping;
    h.getClearColor(lh), h.toneMapping = xi, h.autoClear = !1, h.state.buffers.depth.getReversed() && (h.setRenderTarget(s), h.clearDepth(), h.setRenderTarget(null));
    const x = new Rn({
      name: "PMREM.Background",
      side: Wt,
      depthWrite: !1,
      depthTest: !1
    }), m = new vt(new ji(), x);
    let d = !1;
    const b = e.background;
    b ? b.isColor && (x.color.copy(b), e.background = null, d = !0) : (x.color.copy(lh), d = !0);
    for (let A = 0; A < 6; A++) {
      const M = A % 3;
      M === 0 ? (l.up.set(0, c[A], 0), l.position.set(r.x, r.y, r.z), l.lookAt(r.x + u[A], r.y, r.z)) : M === 1 ? (l.up.set(0, 0, c[A]), l.position.set(r.x, r.y, r.z), l.lookAt(r.x, r.y + u[A], r.z)) : (l.up.set(0, c[A], 0), l.position.set(r.x, r.y, r.z), l.lookAt(r.x, r.y, r.z + u[A]));
      const R = this._cubeSize;
      po(s, M * R, A > 2 ? R : 0, R, R), h.setRenderTarget(s), d && h.render(m, l), h.render(e, l);
    }
    m.geometry.dispose(), m.material.dispose(), h.toneMapping = p, h.autoClear = f, e.background = b;
  }
  _textureToCubeUV(e, t) {
    const i = this._renderer, s = e.mapping === Ns || e.mapping === Fs;
    s ? (this._cubemapMaterial === null && (this._cubemapMaterial = dh()), this._cubemapMaterial.uniforms.flipEnvMap.value = e.isRenderTargetTexture === !1 ? -1 : 1) : this._equirectMaterial === null && (this._equirectMaterial = fh());
    const r = s ? this._cubemapMaterial : this._equirectMaterial, o = new vt(this._lodPlanes[0], r), a = r.uniforms;
    a.envMap.value = e;
    const l = this._cubeSize;
    po(t, 0, 0, 3 * l, 2 * l), i.setRenderTarget(t), i.render(o, qa);
  }
  _applyPMREM(e) {
    const t = this._renderer, i = t.autoClear;
    t.autoClear = !1;
    const s = this._lodPlanes.length;
    for (let r = 1; r < s; r++) {
      const o = Math.sqrt(this._sigmas[r] * this._sigmas[r] - this._sigmas[r - 1] * this._sigmas[r - 1]), a = ch[(s - r - 1) % ch.length];
      this._blur(e, r - 1, r, o, a);
    }
    t.autoClear = i;
  }
  /**
   * This is a two-pass Gaussian blur for a cubemap. Normally this is done
   * vertically and horizontally, but this breaks down on a cube. Here we apply
   * the blur latitudinally (around the poles), and then longitudinally (towards
   * the poles) to approximate the orthogonally-separable blur. It is least
   * accurate at the poles, but still does a decent job.
   *
   * @private
   * @param {WebGLRenderTarget} cubeUVRenderTarget
   * @param {number} lodIn
   * @param {number} lodOut
   * @param {number} sigma
   * @param {Vector3} [poleAxis]
   */
  _blur(e, t, i, s, r) {
    const o = this._pingPongRenderTarget;
    this._halfBlur(
      e,
      o,
      t,
      i,
      s,
      "latitudinal",
      r
    ), this._halfBlur(
      o,
      e,
      i,
      i,
      s,
      "longitudinal",
      r
    );
  }
  _halfBlur(e, t, i, s, r, o, a) {
    const l = this._renderer, c = this._blurMaterial;
    o !== "latitudinal" && o !== "longitudinal" && console.error(
      "blur direction must be either latitudinal or longitudinal!"
    );
    const u = 3, h = new vt(this._lodPlanes[s], c), f = c.uniforms, p = this._sizeLods[i] - 1, v = isFinite(r) ? Math.PI / (2 * p) : 2 * Math.PI / (2 * zi - 1), x = r / v, m = isFinite(r) ? 1 + Math.floor(u * x) : zi;
    m > zi && console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${zi}`);
    const d = [];
    let b = 0;
    for (let D = 0; D < zi; ++D) {
      const U = D / x, y = Math.exp(-U * U / 2);
      d.push(y), D === 0 ? b += y : D < m && (b += 2 * y);
    }
    for (let D = 0; D < d.length; D++)
      d[D] = d[D] / b;
    f.envMap.value = e.texture, f.samples.value = m, f.weights.value = d, f.latitudinal.value = o === "latitudinal", a && (f.poleAxis.value = a);
    const { _lodMax: A } = this;
    f.dTheta.value = v, f.mipInt.value = A - i;
    const M = this._sizeLods[s], R = 3 * M * (s > A - Ts ? s - A + Ts : 0), w = 4 * (this._cubeSize - M);
    po(t, R, w, 3 * M, 2 * M), l.setRenderTarget(t), l.render(h, qa);
  }
}
function aM(n) {
  const e = [], t = [], i = [];
  let s = n;
  const r = n - Ts + 1 + ah.length;
  for (let o = 0; o < r; o++) {
    const a = Math.pow(2, s);
    t.push(a);
    let l = 1 / a;
    o > n - Ts ? l = ah[o - n + Ts - 1] : o === 0 && (l = 0), i.push(l);
    const c = 1 / (a - 2), u = -c, h = 1 + c, f = [u, u, h, u, h, h, u, u, h, h, u, h], p = 6, v = 6, x = 3, m = 2, d = 1, b = new Float32Array(x * v * p), A = new Float32Array(m * v * p), M = new Float32Array(d * v * p);
    for (let w = 0; w < p; w++) {
      const D = w % 3 * 2 / 3 - 1, U = w > 2 ? 0 : -1, y = [
        D,
        U,
        0,
        D + 2 / 3,
        U,
        0,
        D + 2 / 3,
        U + 1,
        0,
        D,
        U,
        0,
        D + 2 / 3,
        U + 1,
        0,
        D,
        U + 1,
        0
      ];
      b.set(y, x * v * w), A.set(f, m * v * w);
      const S = [w, w, w, w, w, w];
      M.set(S, d * v * w);
    }
    const R = new Nt();
    R.setAttribute("position", new En(b, x)), R.setAttribute("uv", new En(A, m)), R.setAttribute("faceIndex", new En(M, d)), e.push(R), s > Ts && s--;
  }
  return { lodPlanes: e, sizeLods: t, sigmas: i };
}
function hh(n, e, t) {
  const i = new qi(n, e, t);
  return i.texture.mapping = ta, i.texture.name = "PMREM.cubeUv", i.scissorTest = !0, i;
}
function po(n, e, t, i, s) {
  n.viewport.set(e, t, i, s), n.scissor.set(e, t, i, s);
}
function lM(n, e, t) {
  const i = new Float32Array(zi), s = new N(0, 1, 0);
  return new yi({
    name: "SphericalGaussianBlur",
    defines: {
      n: zi,
      CUBEUV_TEXEL_WIDTH: 1 / e,
      CUBEUV_TEXEL_HEIGHT: 1 / t,
      CUBEUV_MAX_MIP: `${n}.0`
    },
    uniforms: {
      envMap: { value: null },
      samples: { value: 1 },
      weights: { value: i },
      latitudinal: { value: !1 },
      dTheta: { value: 0 },
      mipInt: { value: 0 },
      poleAxis: { value: s }
    },
    vertexShader: Lc(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`
    ),
    blending: vi,
    depthTest: !1,
    depthWrite: !1
  });
}
function fh() {
  return new yi({
    name: "EquirectangularToCubeUV",
    uniforms: {
      envMap: { value: null }
    },
    vertexShader: Lc(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`
    ),
    blending: vi,
    depthTest: !1,
    depthWrite: !1
  });
}
function dh() {
  return new yi({
    name: "CubemapToCubeUV",
    uniforms: {
      envMap: { value: null },
      flipEnvMap: { value: -1 }
    },
    vertexShader: Lc(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`
    ),
    blending: vi,
    depthTest: !1,
    depthWrite: !1
  });
}
function Lc() {
  return (
    /* glsl */
    `

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`
  );
}
function cM(n) {
  let e = /* @__PURE__ */ new WeakMap(), t = null;
  function i(a) {
    if (a && a.isTexture) {
      const l = a.mapping, c = l === Sl || l === yl, u = l === Ns || l === Fs;
      if (c || u) {
        let h = e.get(a);
        const f = h !== void 0 ? h.texture.pmremVersion : 0;
        if (a.isRenderTargetTexture && a.pmremVersion !== f)
          return t === null && (t = new uh(n)), h = c ? t.fromEquirectangular(a, h) : t.fromCubemap(a, h), h.texture.pmremVersion = a.pmremVersion, e.set(a, h), h.texture;
        if (h !== void 0)
          return h.texture;
        {
          const p = a.image;
          return c && p && p.height > 0 || u && p && s(p) ? (t === null && (t = new uh(n)), h = c ? t.fromEquirectangular(a) : t.fromCubemap(a), h.texture.pmremVersion = a.pmremVersion, e.set(a, h), a.addEventListener("dispose", r), h.texture) : null;
        }
      }
    }
    return a;
  }
  function s(a) {
    let l = 0;
    const c = 6;
    for (let u = 0; u < c; u++)
      a[u] !== void 0 && l++;
    return l === c;
  }
  function r(a) {
    const l = a.target;
    l.removeEventListener("dispose", r);
    const c = e.get(l);
    c !== void 0 && (e.delete(l), c.dispose());
  }
  function o() {
    e = /* @__PURE__ */ new WeakMap(), t !== null && (t.dispose(), t = null);
  }
  return {
    get: i,
    dispose: o
  };
}
function uM(n) {
  const e = {};
  function t(i) {
    if (e[i] !== void 0)
      return e[i];
    let s;
    switch (i) {
      case "WEBGL_depth_texture":
        s = n.getExtension("WEBGL_depth_texture") || n.getExtension("MOZ_WEBGL_depth_texture") || n.getExtension("WEBKIT_WEBGL_depth_texture");
        break;
      case "EXT_texture_filter_anisotropic":
        s = n.getExtension("EXT_texture_filter_anisotropic") || n.getExtension("MOZ_EXT_texture_filter_anisotropic") || n.getExtension("WEBKIT_EXT_texture_filter_anisotropic");
        break;
      case "WEBGL_compressed_texture_s3tc":
        s = n.getExtension("WEBGL_compressed_texture_s3tc") || n.getExtension("MOZ_WEBGL_compressed_texture_s3tc") || n.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");
        break;
      case "WEBGL_compressed_texture_pvrtc":
        s = n.getExtension("WEBGL_compressed_texture_pvrtc") || n.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");
        break;
      default:
        s = n.getExtension(i);
    }
    return e[i] = s, s;
  }
  return {
    has: function(i) {
      return t(i) !== null;
    },
    init: function() {
      t("EXT_color_buffer_float"), t("WEBGL_clip_cull_distance"), t("OES_texture_float_linear"), t("EXT_color_buffer_half_float"), t("WEBGL_multisampled_render_to_texture"), t("WEBGL_render_shared_exponent");
    },
    get: function(i) {
      const s = t(i);
      return s === null && Rr("THREE.WebGLRenderer: " + i + " extension not supported."), s;
    }
  };
}
function hM(n, e, t, i) {
  const s = {}, r = /* @__PURE__ */ new WeakMap();
  function o(h) {
    const f = h.target;
    f.index !== null && e.remove(f.index);
    for (const v in f.attributes)
      e.remove(f.attributes[v]);
    f.removeEventListener("dispose", o), delete s[f.id];
    const p = r.get(f);
    p && (e.remove(p), r.delete(f)), i.releaseStatesOfGeometry(f), f.isInstancedBufferGeometry === !0 && delete f._maxInstanceCount, t.memory.geometries--;
  }
  function a(h, f) {
    return s[f.id] === !0 || (f.addEventListener("dispose", o), s[f.id] = !0, t.memory.geometries++), f;
  }
  function l(h) {
    const f = h.attributes;
    for (const p in f)
      e.update(f[p], n.ARRAY_BUFFER);
  }
  function c(h) {
    const f = [], p = h.index, v = h.attributes.position;
    let x = 0;
    if (p !== null) {
      const b = p.array;
      x = p.version;
      for (let A = 0, M = b.length; A < M; A += 3) {
        const R = b[A + 0], w = b[A + 1], D = b[A + 2];
        f.push(R, w, w, D, D, R);
      }
    } else if (v !== void 0) {
      const b = v.array;
      x = v.version;
      for (let A = 0, M = b.length / 3 - 1; A < M; A += 3) {
        const R = A + 0, w = A + 1, D = A + 2;
        f.push(R, w, w, D, D, R);
      }
    } else
      return;
    const m = new (fd(f) ? gd : _d)(f, 1);
    m.version = x;
    const d = r.get(h);
    d && e.remove(d), r.set(h, m);
  }
  function u(h) {
    const f = r.get(h);
    if (f) {
      const p = h.index;
      p !== null && f.version < p.version && c(h);
    } else
      c(h);
    return r.get(h);
  }
  return {
    get: a,
    update: l,
    getWireframeAttribute: u
  };
}
function fM(n, e, t) {
  let i;
  function s(f) {
    i = f;
  }
  let r, o;
  function a(f) {
    r = f.type, o = f.bytesPerElement;
  }
  function l(f, p) {
    n.drawElements(i, p, r, f * o), t.update(p, i, 1);
  }
  function c(f, p, v) {
    v !== 0 && (n.drawElementsInstanced(i, p, r, f * o, v), t.update(p, i, v));
  }
  function u(f, p, v) {
    if (v === 0) return;
    e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i, p, 0, r, f, 0, v);
    let m = 0;
    for (let d = 0; d < v; d++)
      m += p[d];
    t.update(m, i, 1);
  }
  function h(f, p, v, x) {
    if (v === 0) return;
    const m = e.get("WEBGL_multi_draw");
    if (m === null)
      for (let d = 0; d < f.length; d++)
        c(f[d] / o, p[d], x[d]);
    else {
      m.multiDrawElementsInstancedWEBGL(i, p, 0, r, f, 0, x, 0, v);
      let d = 0;
      for (let b = 0; b < v; b++)
        d += p[b] * x[b];
      t.update(d, i, 1);
    }
  }
  this.setMode = s, this.setIndex = a, this.render = l, this.renderInstances = c, this.renderMultiDraw = u, this.renderMultiDrawInstances = h;
}
function dM(n) {
  const e = {
    geometries: 0,
    textures: 0
  }, t = {
    frame: 0,
    calls: 0,
    triangles: 0,
    points: 0,
    lines: 0
  };
  function i(r, o, a) {
    switch (t.calls++, o) {
      case n.TRIANGLES:
        t.triangles += a * (r / 3);
        break;
      case n.LINES:
        t.lines += a * (r / 2);
        break;
      case n.LINE_STRIP:
        t.lines += a * (r - 1);
        break;
      case n.LINE_LOOP:
        t.lines += a * r;
        break;
      case n.POINTS:
        t.points += a * r;
        break;
      default:
        console.error("THREE.WebGLInfo: Unknown draw mode:", o);
        break;
    }
  }
  function s() {
    t.calls = 0, t.triangles = 0, t.points = 0, t.lines = 0;
  }
  return {
    memory: e,
    render: t,
    programs: null,
    autoReset: !0,
    reset: s,
    update: i
  };
}
function pM(n, e, t) {
  const i = /* @__PURE__ */ new WeakMap(), s = new lt();
  function r(o, a, l) {
    const c = o.morphTargetInfluences, u = a.morphAttributes.position || a.morphAttributes.normal || a.morphAttributes.color, h = u !== void 0 ? u.length : 0;
    let f = i.get(a);
    if (f === void 0 || f.count !== h) {
      let y = function() {
        D.dispose(), i.delete(a), a.removeEventListener("dispose", y);
      };
      f !== void 0 && f.texture.dispose();
      const p = a.morphAttributes.position !== void 0, v = a.morphAttributes.normal !== void 0, x = a.morphAttributes.color !== void 0, m = a.morphAttributes.position || [], d = a.morphAttributes.normal || [], b = a.morphAttributes.color || [];
      let A = 0;
      p === !0 && (A = 1), v === !0 && (A = 2), x === !0 && (A = 3);
      let M = a.attributes.position.count * A, R = 1;
      M > e.maxTextureSize && (R = Math.ceil(M / e.maxTextureSize), M = e.maxTextureSize);
      const w = new Float32Array(M * R * 4 * h), D = new dd(w, M, R, h);
      D.type = ei, D.needsUpdate = !0;
      const U = A * 4;
      for (let S = 0; S < h; S++) {
        const P = m[S], L = d[S], V = b[S], Z = M * R * 4 * S;
        for (let te = 0; te < P.count; te++) {
          const $ = te * U;
          p === !0 && (s.fromBufferAttribute(P, te), w[Z + $ + 0] = s.x, w[Z + $ + 1] = s.y, w[Z + $ + 2] = s.z, w[Z + $ + 3] = 0), v === !0 && (s.fromBufferAttribute(L, te), w[Z + $ + 4] = s.x, w[Z + $ + 5] = s.y, w[Z + $ + 6] = s.z, w[Z + $ + 7] = 0), x === !0 && (s.fromBufferAttribute(V, te), w[Z + $ + 8] = s.x, w[Z + $ + 9] = s.y, w[Z + $ + 10] = s.z, w[Z + $ + 11] = V.itemSize === 4 ? s.w : 1);
        }
      }
      f = {
        count: h,
        texture: D,
        size: new Ve(M, R)
      }, i.set(a, f), a.addEventListener("dispose", y);
    }
    if (o.isInstancedMesh === !0 && o.morphTexture !== null)
      l.getUniforms().setValue(n, "morphTexture", o.morphTexture, t);
    else {
      let p = 0;
      for (let x = 0; x < c.length; x++)
        p += c[x];
      const v = a.morphTargetsRelative ? 1 : 1 - p;
      l.getUniforms().setValue(n, "morphTargetBaseInfluence", v), l.getUniforms().setValue(n, "morphTargetInfluences", c);
    }
    l.getUniforms().setValue(n, "morphTargetsTexture", f.texture, t), l.getUniforms().setValue(n, "morphTargetsTextureSize", f.size);
  }
  return {
    update: r
  };
}
function mM(n, e, t, i) {
  let s = /* @__PURE__ */ new WeakMap();
  function r(l) {
    const c = i.render.frame, u = l.geometry, h = e.get(l, u);
    if (s.get(h) !== c && (e.update(h), s.set(h, c)), l.isInstancedMesh && (l.hasEventListener("dispose", a) === !1 && l.addEventListener("dispose", a), s.get(l) !== c && (t.update(l.instanceMatrix, n.ARRAY_BUFFER), l.instanceColor !== null && t.update(l.instanceColor, n.ARRAY_BUFFER), s.set(l, c))), l.isSkinnedMesh) {
      const f = l.skeleton;
      s.get(f) !== c && (f.update(), s.set(f, c));
    }
    return h;
  }
  function o() {
    s = /* @__PURE__ */ new WeakMap();
  }
  function a(l) {
    const c = l.target;
    c.removeEventListener("dispose", a), t.remove(c.instanceMatrix), c.instanceColor !== null && t.remove(c.instanceColor);
  }
  return {
    update: r,
    dispose: o
  };
}
const Rd = /* @__PURE__ */ new $t(), ph = /* @__PURE__ */ new Ed(1, 1), Cd = /* @__PURE__ */ new dd(), Pd = /* @__PURE__ */ new Fg(), Dd = /* @__PURE__ */ new Md(), mh = [], _h = [], gh = new Float32Array(16), vh = new Float32Array(9), xh = new Float32Array(4);
function Vs(n, e, t) {
  const i = n[0];
  if (i <= 0 || i > 0) return n;
  const s = e * t;
  let r = mh[s];
  if (r === void 0 && (r = new Float32Array(s), mh[s] = r), e !== 0) {
    i.toArray(r, 0);
    for (let o = 1, a = 0; o !== e; ++o)
      a += t, n[o].toArray(r, a);
  }
  return r;
}
function bt(n, e) {
  if (n.length !== e.length) return !1;
  for (let t = 0, i = n.length; t < i; t++)
    if (n[t] !== e[t]) return !1;
  return !0;
}
function At(n, e) {
  for (let t = 0, i = e.length; t < i; t++)
    n[t] = e[t];
}
function ia(n, e) {
  let t = _h[e];
  t === void 0 && (t = new Int32Array(e), _h[e] = t);
  for (let i = 0; i !== e; ++i)
    t[i] = n.allocateTextureUnit();
  return t;
}
function _M(n, e) {
  const t = this.cache;
  t[0] !== e && (n.uniform1f(this.addr, e), t[0] = e);
}
function gM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y) && (n.uniform2f(this.addr, e.x, e.y), t[0] = e.x, t[1] = e.y);
  else {
    if (bt(t, e)) return;
    n.uniform2fv(this.addr, e), At(t, e);
  }
}
function vM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z) && (n.uniform3f(this.addr, e.x, e.y, e.z), t[0] = e.x, t[1] = e.y, t[2] = e.z);
  else if (e.r !== void 0)
    (t[0] !== e.r || t[1] !== e.g || t[2] !== e.b) && (n.uniform3f(this.addr, e.r, e.g, e.b), t[0] = e.r, t[1] = e.g, t[2] = e.b);
  else {
    if (bt(t, e)) return;
    n.uniform3fv(this.addr, e), At(t, e);
  }
}
function xM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z || t[3] !== e.w) && (n.uniform4f(this.addr, e.x, e.y, e.z, e.w), t[0] = e.x, t[1] = e.y, t[2] = e.z, t[3] = e.w);
  else {
    if (bt(t, e)) return;
    n.uniform4fv(this.addr, e), At(t, e);
  }
}
function MM(n, e) {
  const t = this.cache, i = e.elements;
  if (i === void 0) {
    if (bt(t, e)) return;
    n.uniformMatrix2fv(this.addr, !1, e), At(t, e);
  } else {
    if (bt(t, i)) return;
    xh.set(i), n.uniformMatrix2fv(this.addr, !1, xh), At(t, i);
  }
}
function SM(n, e) {
  const t = this.cache, i = e.elements;
  if (i === void 0) {
    if (bt(t, e)) return;
    n.uniformMatrix3fv(this.addr, !1, e), At(t, e);
  } else {
    if (bt(t, i)) return;
    vh.set(i), n.uniformMatrix3fv(this.addr, !1, vh), At(t, i);
  }
}
function yM(n, e) {
  const t = this.cache, i = e.elements;
  if (i === void 0) {
    if (bt(t, e)) return;
    n.uniformMatrix4fv(this.addr, !1, e), At(t, e);
  } else {
    if (bt(t, i)) return;
    gh.set(i), n.uniformMatrix4fv(this.addr, !1, gh), At(t, i);
  }
}
function EM(n, e) {
  const t = this.cache;
  t[0] !== e && (n.uniform1i(this.addr, e), t[0] = e);
}
function TM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y) && (n.uniform2i(this.addr, e.x, e.y), t[0] = e.x, t[1] = e.y);
  else {
    if (bt(t, e)) return;
    n.uniform2iv(this.addr, e), At(t, e);
  }
}
function bM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z) && (n.uniform3i(this.addr, e.x, e.y, e.z), t[0] = e.x, t[1] = e.y, t[2] = e.z);
  else {
    if (bt(t, e)) return;
    n.uniform3iv(this.addr, e), At(t, e);
  }
}
function AM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z || t[3] !== e.w) && (n.uniform4i(this.addr, e.x, e.y, e.z, e.w), t[0] = e.x, t[1] = e.y, t[2] = e.z, t[3] = e.w);
  else {
    if (bt(t, e)) return;
    n.uniform4iv(this.addr, e), At(t, e);
  }
}
function wM(n, e) {
  const t = this.cache;
  t[0] !== e && (n.uniform1ui(this.addr, e), t[0] = e);
}
function RM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y) && (n.uniform2ui(this.addr, e.x, e.y), t[0] = e.x, t[1] = e.y);
  else {
    if (bt(t, e)) return;
    n.uniform2uiv(this.addr, e), At(t, e);
  }
}
function CM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z) && (n.uniform3ui(this.addr, e.x, e.y, e.z), t[0] = e.x, t[1] = e.y, t[2] = e.z);
  else {
    if (bt(t, e)) return;
    n.uniform3uiv(this.addr, e), At(t, e);
  }
}
function PM(n, e) {
  const t = this.cache;
  if (e.x !== void 0)
    (t[0] !== e.x || t[1] !== e.y || t[2] !== e.z || t[3] !== e.w) && (n.uniform4ui(this.addr, e.x, e.y, e.z, e.w), t[0] = e.x, t[1] = e.y, t[2] = e.z, t[3] = e.w);
  else {
    if (bt(t, e)) return;
    n.uniform4uiv(this.addr, e), At(t, e);
  }
}
function DM(n, e, t) {
  const i = this.cache, s = t.allocateTextureUnit();
  i[0] !== s && (n.uniform1i(this.addr, s), i[0] = s);
  let r;
  this.type === n.SAMPLER_2D_SHADOW ? (ph.compareFunction = hd, r = ph) : r = Rd, t.setTexture2D(e || r, s);
}
function LM(n, e, t) {
  const i = this.cache, s = t.allocateTextureUnit();
  i[0] !== s && (n.uniform1i(this.addr, s), i[0] = s), t.setTexture3D(e || Pd, s);
}
function IM(n, e, t) {
  const i = this.cache, s = t.allocateTextureUnit();
  i[0] !== s && (n.uniform1i(this.addr, s), i[0] = s), t.setTextureCube(e || Dd, s);
}
function UM(n, e, t) {
  const i = this.cache, s = t.allocateTextureUnit();
  i[0] !== s && (n.uniform1i(this.addr, s), i[0] = s), t.setTexture2DArray(e || Cd, s);
}
function NM(n) {
  switch (n) {
    case 5126:
      return _M;
    // FLOAT
    case 35664:
      return gM;
    // _VEC2
    case 35665:
      return vM;
    // _VEC3
    case 35666:
      return xM;
    // _VEC4
    case 35674:
      return MM;
    // _MAT2
    case 35675:
      return SM;
    // _MAT3
    case 35676:
      return yM;
    // _MAT4
    case 5124:
    case 35670:
      return EM;
    // INT, BOOL
    case 35667:
    case 35671:
      return TM;
    // _VEC2
    case 35668:
    case 35672:
      return bM;
    // _VEC3
    case 35669:
    case 35673:
      return AM;
    // _VEC4
    case 5125:
      return wM;
    // UINT
    case 36294:
      return RM;
    // _VEC2
    case 36295:
      return CM;
    // _VEC3
    case 36296:
      return PM;
    // _VEC4
    case 35678:
    // SAMPLER_2D
    case 36198:
    // SAMPLER_EXTERNAL_OES
    case 36298:
    // INT_SAMPLER_2D
    case 36306:
    // UNSIGNED_INT_SAMPLER_2D
    case 35682:
      return DM;
    case 35679:
    // SAMPLER_3D
    case 36299:
    // INT_SAMPLER_3D
    case 36307:
      return LM;
    case 35680:
    // SAMPLER_CUBE
    case 36300:
    // INT_SAMPLER_CUBE
    case 36308:
    // UNSIGNED_INT_SAMPLER_CUBE
    case 36293:
      return IM;
    case 36289:
    // SAMPLER_2D_ARRAY
    case 36303:
    // INT_SAMPLER_2D_ARRAY
    case 36311:
    // UNSIGNED_INT_SAMPLER_2D_ARRAY
    case 36292:
      return UM;
  }
}
function FM(n, e) {
  n.uniform1fv(this.addr, e);
}
function OM(n, e) {
  const t = Vs(e, this.size, 2);
  n.uniform2fv(this.addr, t);
}
function BM(n, e) {
  const t = Vs(e, this.size, 3);
  n.uniform3fv(this.addr, t);
}
function zM(n, e) {
  const t = Vs(e, this.size, 4);
  n.uniform4fv(this.addr, t);
}
function HM(n, e) {
  const t = Vs(e, this.size, 4);
  n.uniformMatrix2fv(this.addr, !1, t);
}
function VM(n, e) {
  const t = Vs(e, this.size, 9);
  n.uniformMatrix3fv(this.addr, !1, t);
}
function kM(n, e) {
  const t = Vs(e, this.size, 16);
  n.uniformMatrix4fv(this.addr, !1, t);
}
function GM(n, e) {
  n.uniform1iv(this.addr, e);
}
function WM(n, e) {
  n.uniform2iv(this.addr, e);
}
function XM(n, e) {
  n.uniform3iv(this.addr, e);
}
function YM(n, e) {
  n.uniform4iv(this.addr, e);
}
function qM(n, e) {
  n.uniform1uiv(this.addr, e);
}
function jM(n, e) {
  n.uniform2uiv(this.addr, e);
}
function KM(n, e) {
  n.uniform3uiv(this.addr, e);
}
function $M(n, e) {
  n.uniform4uiv(this.addr, e);
}
function ZM(n, e, t) {
  const i = this.cache, s = e.length, r = ia(t, s);
  bt(i, r) || (n.uniform1iv(this.addr, r), At(i, r));
  for (let o = 0; o !== s; ++o)
    t.setTexture2D(e[o] || Rd, r[o]);
}
function JM(n, e, t) {
  const i = this.cache, s = e.length, r = ia(t, s);
  bt(i, r) || (n.uniform1iv(this.addr, r), At(i, r));
  for (let o = 0; o !== s; ++o)
    t.setTexture3D(e[o] || Pd, r[o]);
}
function QM(n, e, t) {
  const i = this.cache, s = e.length, r = ia(t, s);
  bt(i, r) || (n.uniform1iv(this.addr, r), At(i, r));
  for (let o = 0; o !== s; ++o)
    t.setTextureCube(e[o] || Dd, r[o]);
}
function eS(n, e, t) {
  const i = this.cache, s = e.length, r = ia(t, s);
  bt(i, r) || (n.uniform1iv(this.addr, r), At(i, r));
  for (let o = 0; o !== s; ++o)
    t.setTexture2DArray(e[o] || Cd, r[o]);
}
function tS(n) {
  switch (n) {
    case 5126:
      return FM;
    // FLOAT
    case 35664:
      return OM;
    // _VEC2
    case 35665:
      return BM;
    // _VEC3
    case 35666:
      return zM;
    // _VEC4
    case 35674:
      return HM;
    // _MAT2
    case 35675:
      return VM;
    // _MAT3
    case 35676:
      return kM;
    // _MAT4
    case 5124:
    case 35670:
      return GM;
    // INT, BOOL
    case 35667:
    case 35671:
      return WM;
    // _VEC2
    case 35668:
    case 35672:
      return XM;
    // _VEC3
    case 35669:
    case 35673:
      return YM;
    // _VEC4
    case 5125:
      return qM;
    // UINT
    case 36294:
      return jM;
    // _VEC2
    case 36295:
      return KM;
    // _VEC3
    case 36296:
      return $M;
    // _VEC4
    case 35678:
    // SAMPLER_2D
    case 36198:
    // SAMPLER_EXTERNAL_OES
    case 36298:
    // INT_SAMPLER_2D
    case 36306:
    // UNSIGNED_INT_SAMPLER_2D
    case 35682:
      return ZM;
    case 35679:
    // SAMPLER_3D
    case 36299:
    // INT_SAMPLER_3D
    case 36307:
      return JM;
    case 35680:
    // SAMPLER_CUBE
    case 36300:
    // INT_SAMPLER_CUBE
    case 36308:
    // UNSIGNED_INT_SAMPLER_CUBE
    case 36293:
      return QM;
    case 36289:
    // SAMPLER_2D_ARRAY
    case 36303:
    // INT_SAMPLER_2D_ARRAY
    case 36311:
    // UNSIGNED_INT_SAMPLER_2D_ARRAY
    case 36292:
      return eS;
  }
}
class nS {
  constructor(e, t, i) {
    this.id = e, this.addr = i, this.cache = [], this.type = t.type, this.setValue = NM(t.type);
  }
}
class iS {
  constructor(e, t, i) {
    this.id = e, this.addr = i, this.cache = [], this.type = t.type, this.size = t.size, this.setValue = tS(t.type);
  }
}
class sS {
  constructor(e) {
    this.id = e, this.seq = [], this.map = {};
  }
  setValue(e, t, i) {
    const s = this.seq;
    for (let r = 0, o = s.length; r !== o; ++r) {
      const a = s[r];
      a.setValue(e, t[a.id], i);
    }
  }
}
const Ja = /(\w+)(\])?(\[|\.)?/g;
function Mh(n, e) {
  n.seq.push(e), n.map[e.id] = e;
}
function rS(n, e, t) {
  const i = n.name, s = i.length;
  for (Ja.lastIndex = 0; ; ) {
    const r = Ja.exec(i), o = Ja.lastIndex;
    let a = r[1];
    const l = r[2] === "]", c = r[3];
    if (l && (a = a | 0), c === void 0 || c === "[" && o + 2 === s) {
      Mh(t, c === void 0 ? new nS(a, n, e) : new iS(a, n, e));
      break;
    } else {
      let h = t.map[a];
      h === void 0 && (h = new sS(a), Mh(t, h)), t = h;
    }
  }
}
class Ao {
  constructor(e, t) {
    this.seq = [], this.map = {};
    const i = e.getProgramParameter(t, e.ACTIVE_UNIFORMS);
    for (let s = 0; s < i; ++s) {
      const r = e.getActiveUniform(t, s), o = e.getUniformLocation(t, r.name);
      rS(r, o, this);
    }
  }
  setValue(e, t, i, s) {
    const r = this.map[t];
    r !== void 0 && r.setValue(e, i, s);
  }
  setOptional(e, t, i) {
    const s = t[i];
    s !== void 0 && this.setValue(e, i, s);
  }
  static upload(e, t, i, s) {
    for (let r = 0, o = t.length; r !== o; ++r) {
      const a = t[r], l = i[a.id];
      l.needsUpdate !== !1 && a.setValue(e, l.value, s);
    }
  }
  static seqWithValue(e, t) {
    const i = [];
    for (let s = 0, r = e.length; s !== r; ++s) {
      const o = e[s];
      o.id in t && i.push(o);
    }
    return i;
  }
}
function Sh(n, e, t) {
  const i = n.createShader(e);
  return n.shaderSource(i, t), n.compileShader(i), i;
}
const oS = 37297;
let aS = 0;
function lS(n, e) {
  const t = n.split(`
`), i = [], s = Math.max(e - 6, 0), r = Math.min(e + 6, t.length);
  for (let o = s; o < r; o++) {
    const a = o + 1;
    i.push(`${a === e ? ">" : " "} ${a}: ${t[o]}`);
  }
  return i.join(`
`);
}
const yh = /* @__PURE__ */ new Ye();
function cS(n) {
  Qe._getMatrix(yh, Qe.workingColorSpace, n);
  const e = `mat3( ${yh.elements.map((t) => t.toFixed(4))} )`;
  switch (Qe.getTransfer(n)) {
    case Bo:
      return [e, "LinearTransferOETF"];
    case ot:
      return [e, "sRGBTransferOETF"];
    default:
      return console.warn("THREE.WebGLProgram: Unsupported color space: ", n), [e, "LinearTransferOETF"];
  }
}
function Eh(n, e, t) {
  const i = n.getShaderParameter(e, n.COMPILE_STATUS), r = (n.getShaderInfoLog(e) || "").trim();
  if (i && r === "") return "";
  const o = /ERROR: 0:(\d+)/.exec(r);
  if (o) {
    const a = parseInt(o[1]);
    return t.toUpperCase() + `

` + r + `

` + lS(n.getShaderSource(e), a);
  } else
    return r;
}
function uS(n, e) {
  const t = cS(e);
  return [
    `vec4 ${n}( vec4 value ) {`,
    `	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,
    "}"
  ].join(`
`);
}
function hS(n, e) {
  let t;
  switch (e) {
    case ug:
      t = "Linear";
      break;
    case hg:
      t = "Reinhard";
      break;
    case fg:
      t = "Cineon";
      break;
    case ed:
      t = "ACESFilmic";
      break;
    case pg:
      t = "AgX";
      break;
    case mg:
      t = "Neutral";
      break;
    case dg:
      t = "Custom";
      break;
    default:
      console.warn("THREE.WebGLProgram: Unsupported toneMapping:", e), t = "Linear";
  }
  return "vec3 " + n + "( vec3 color ) { return " + t + "ToneMapping( color ); }";
}
const mo = /* @__PURE__ */ new N();
function fS() {
  Qe.getLuminanceCoefficients(mo);
  const n = mo.x.toFixed(4), e = mo.y.toFixed(4), t = mo.z.toFixed(4);
  return [
    "float luminance( const in vec3 rgb ) {",
    `	const vec3 weights = vec3( ${n}, ${e}, ${t} );`,
    "	return dot( weights, rgb );",
    "}"
  ].join(`
`);
}
function dS(n) {
  return [
    n.extensionClipCullDistance ? "#extension GL_ANGLE_clip_cull_distance : require" : "",
    n.extensionMultiDraw ? "#extension GL_ANGLE_multi_draw : require" : ""
  ].filter(or).join(`
`);
}
function pS(n) {
  const e = [];
  for (const t in n) {
    const i = n[t];
    i !== !1 && e.push("#define " + t + " " + i);
  }
  return e.join(`
`);
}
function mS(n, e) {
  const t = {}, i = n.getProgramParameter(e, n.ACTIVE_ATTRIBUTES);
  for (let s = 0; s < i; s++) {
    const r = n.getActiveAttrib(e, s), o = r.name;
    let a = 1;
    r.type === n.FLOAT_MAT2 && (a = 2), r.type === n.FLOAT_MAT3 && (a = 3), r.type === n.FLOAT_MAT4 && (a = 4), t[o] = {
      type: r.type,
      location: n.getAttribLocation(e, o),
      locationSize: a
    };
  }
  return t;
}
function or(n) {
  return n !== "";
}
function Th(n, e) {
  const t = e.numSpotLightShadows + e.numSpotLightMaps - e.numSpotLightShadowsWithMaps;
  return n.replace(/NUM_DIR_LIGHTS/g, e.numDirLights).replace(/NUM_SPOT_LIGHTS/g, e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g, e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g, t).replace(/NUM_RECT_AREA_LIGHTS/g, e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g, e.numPointLights).replace(/NUM_HEMI_LIGHTS/g, e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g, e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g, e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g, e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g, e.numPointLightShadows);
}
function bh(n, e) {
  return n.replace(/NUM_CLIPPING_PLANES/g, e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g, e.numClippingPlanes - e.numClipIntersection);
}
const _S = /^[ \t]*#include +<([\w\d./]+)>/gm;
function tc(n) {
  return n.replace(_S, vS);
}
const gS = /* @__PURE__ */ new Map();
function vS(n, e) {
  let t = qe[e];
  if (t === void 0) {
    const i = gS.get(e);
    if (i !== void 0)
      t = qe[i], console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.', e, i);
    else
      throw new Error("Can not resolve #include <" + e + ">");
  }
  return tc(t);
}
const xS = /#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;
function Ah(n) {
  return n.replace(xS, MS);
}
function MS(n, e, t, i) {
  let s = "";
  for (let r = parseInt(e); r < parseInt(t); r++)
    s += i.replace(/\[\s*i\s*\]/g, "[ " + r + " ]").replace(/UNROLLED_LOOP_INDEX/g, r);
  return s;
}
function wh(n) {
  let e = `precision ${n.precision} float;
	precision ${n.precision} int;
	precision ${n.precision} sampler2D;
	precision ${n.precision} samplerCube;
	precision ${n.precision} sampler3D;
	precision ${n.precision} sampler2DArray;
	precision ${n.precision} sampler2DShadow;
	precision ${n.precision} samplerCubeShadow;
	precision ${n.precision} sampler2DArrayShadow;
	precision ${n.precision} isampler2D;
	precision ${n.precision} isampler3D;
	precision ${n.precision} isamplerCube;
	precision ${n.precision} isampler2DArray;
	precision ${n.precision} usampler2D;
	precision ${n.precision} usampler3D;
	precision ${n.precision} usamplerCube;
	precision ${n.precision} usampler2DArray;
	`;
  return n.precision === "highp" ? e += `
#define HIGH_PRECISION` : n.precision === "mediump" ? e += `
#define MEDIUM_PRECISION` : n.precision === "lowp" && (e += `
#define LOW_PRECISION`), e;
}
function SS(n) {
  let e = "SHADOWMAP_TYPE_BASIC";
  return n.shadowMapType === Jf ? e = "SHADOWMAP_TYPE_PCF" : n.shadowMapType === G_ ? e = "SHADOWMAP_TYPE_PCF_SOFT" : n.shadowMapType === jn && (e = "SHADOWMAP_TYPE_VSM"), e;
}
function yS(n) {
  let e = "ENVMAP_TYPE_CUBE";
  if (n.envMap)
    switch (n.envMapMode) {
      case Ns:
      case Fs:
        e = "ENVMAP_TYPE_CUBE";
        break;
      case ta:
        e = "ENVMAP_TYPE_CUBE_UV";
        break;
    }
  return e;
}
function ES(n) {
  let e = "ENVMAP_MODE_REFLECTION";
  return n.envMap && n.envMapMode === Fs && (e = "ENVMAP_MODE_REFRACTION"), e;
}
function TS(n) {
  let e = "ENVMAP_BLENDING_NONE";
  if (n.envMap)
    switch (n.combine) {
      case Qf:
        e = "ENVMAP_BLENDING_MULTIPLY";
        break;
      case lg:
        e = "ENVMAP_BLENDING_MIX";
        break;
      case cg:
        e = "ENVMAP_BLENDING_ADD";
        break;
    }
  return e;
}
function bS(n) {
  const e = n.envMapCubeUVHeight;
  if (e === null) return null;
  const t = Math.log2(e) - 2, i = 1 / e;
  return { texelWidth: 1 / (3 * Math.max(Math.pow(2, t), 112)), texelHeight: i, maxMip: t };
}
function AS(n, e, t, i) {
  const s = n.getContext(), r = t.defines;
  let o = t.vertexShader, a = t.fragmentShader;
  const l = SS(t), c = yS(t), u = ES(t), h = TS(t), f = bS(t), p = dS(t), v = pS(r), x = s.createProgram();
  let m, d, b = t.glslVersion ? "#version " + t.glslVersion + `
` : "";
  t.isRawShaderMaterial ? (m = [
    "#define SHADER_TYPE " + t.shaderType,
    "#define SHADER_NAME " + t.shaderName,
    v
  ].filter(or).join(`
`), m.length > 0 && (m += `
`), d = [
    "#define SHADER_TYPE " + t.shaderType,
    "#define SHADER_NAME " + t.shaderName,
    v
  ].filter(or).join(`
`), d.length > 0 && (d += `
`)) : (m = [
    wh(t),
    "#define SHADER_TYPE " + t.shaderType,
    "#define SHADER_NAME " + t.shaderName,
    v,
    t.extensionClipCullDistance ? "#define USE_CLIP_DISTANCE" : "",
    t.batching ? "#define USE_BATCHING" : "",
    t.batchingColor ? "#define USE_BATCHING_COLOR" : "",
    t.instancing ? "#define USE_INSTANCING" : "",
    t.instancingColor ? "#define USE_INSTANCING_COLOR" : "",
    t.instancingMorph ? "#define USE_INSTANCING_MORPH" : "",
    t.useFog && t.fog ? "#define USE_FOG" : "",
    t.useFog && t.fogExp2 ? "#define FOG_EXP2" : "",
    t.map ? "#define USE_MAP" : "",
    t.envMap ? "#define USE_ENVMAP" : "",
    t.envMap ? "#define " + u : "",
    t.lightMap ? "#define USE_LIGHTMAP" : "",
    t.aoMap ? "#define USE_AOMAP" : "",
    t.bumpMap ? "#define USE_BUMPMAP" : "",
    t.normalMap ? "#define USE_NORMALMAP" : "",
    t.normalMapObjectSpace ? "#define USE_NORMALMAP_OBJECTSPACE" : "",
    t.normalMapTangentSpace ? "#define USE_NORMALMAP_TANGENTSPACE" : "",
    t.displacementMap ? "#define USE_DISPLACEMENTMAP" : "",
    t.emissiveMap ? "#define USE_EMISSIVEMAP" : "",
    t.anisotropy ? "#define USE_ANISOTROPY" : "",
    t.anisotropyMap ? "#define USE_ANISOTROPYMAP" : "",
    t.clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
    t.clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
    t.clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
    t.iridescenceMap ? "#define USE_IRIDESCENCEMAP" : "",
    t.iridescenceThicknessMap ? "#define USE_IRIDESCENCE_THICKNESSMAP" : "",
    t.specularMap ? "#define USE_SPECULARMAP" : "",
    t.specularColorMap ? "#define USE_SPECULAR_COLORMAP" : "",
    t.specularIntensityMap ? "#define USE_SPECULAR_INTENSITYMAP" : "",
    t.roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
    t.metalnessMap ? "#define USE_METALNESSMAP" : "",
    t.alphaMap ? "#define USE_ALPHAMAP" : "",
    t.alphaHash ? "#define USE_ALPHAHASH" : "",
    t.transmission ? "#define USE_TRANSMISSION" : "",
    t.transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
    t.thicknessMap ? "#define USE_THICKNESSMAP" : "",
    t.sheenColorMap ? "#define USE_SHEEN_COLORMAP" : "",
    t.sheenRoughnessMap ? "#define USE_SHEEN_ROUGHNESSMAP" : "",
    //
    t.mapUv ? "#define MAP_UV " + t.mapUv : "",
    t.alphaMapUv ? "#define ALPHAMAP_UV " + t.alphaMapUv : "",
    t.lightMapUv ? "#define LIGHTMAP_UV " + t.lightMapUv : "",
    t.aoMapUv ? "#define AOMAP_UV " + t.aoMapUv : "",
    t.emissiveMapUv ? "#define EMISSIVEMAP_UV " + t.emissiveMapUv : "",
    t.bumpMapUv ? "#define BUMPMAP_UV " + t.bumpMapUv : "",
    t.normalMapUv ? "#define NORMALMAP_UV " + t.normalMapUv : "",
    t.displacementMapUv ? "#define DISPLACEMENTMAP_UV " + t.displacementMapUv : "",
    t.metalnessMapUv ? "#define METALNESSMAP_UV " + t.metalnessMapUv : "",
    t.roughnessMapUv ? "#define ROUGHNESSMAP_UV " + t.roughnessMapUv : "",
    t.anisotropyMapUv ? "#define ANISOTROPYMAP_UV " + t.anisotropyMapUv : "",
    t.clearcoatMapUv ? "#define CLEARCOATMAP_UV " + t.clearcoatMapUv : "",
    t.clearcoatNormalMapUv ? "#define CLEARCOAT_NORMALMAP_UV " + t.clearcoatNormalMapUv : "",
    t.clearcoatRoughnessMapUv ? "#define CLEARCOAT_ROUGHNESSMAP_UV " + t.clearcoatRoughnessMapUv : "",
    t.iridescenceMapUv ? "#define IRIDESCENCEMAP_UV " + t.iridescenceMapUv : "",
    t.iridescenceThicknessMapUv ? "#define IRIDESCENCE_THICKNESSMAP_UV " + t.iridescenceThicknessMapUv : "",
    t.sheenColorMapUv ? "#define SHEEN_COLORMAP_UV " + t.sheenColorMapUv : "",
    t.sheenRoughnessMapUv ? "#define SHEEN_ROUGHNESSMAP_UV " + t.sheenRoughnessMapUv : "",
    t.specularMapUv ? "#define SPECULARMAP_UV " + t.specularMapUv : "",
    t.specularColorMapUv ? "#define SPECULAR_COLORMAP_UV " + t.specularColorMapUv : "",
    t.specularIntensityMapUv ? "#define SPECULAR_INTENSITYMAP_UV " + t.specularIntensityMapUv : "",
    t.transmissionMapUv ? "#define TRANSMISSIONMAP_UV " + t.transmissionMapUv : "",
    t.thicknessMapUv ? "#define THICKNESSMAP_UV " + t.thicknessMapUv : "",
    //
    t.vertexTangents && t.flatShading === !1 ? "#define USE_TANGENT" : "",
    t.vertexColors ? "#define USE_COLOR" : "",
    t.vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
    t.vertexUv1s ? "#define USE_UV1" : "",
    t.vertexUv2s ? "#define USE_UV2" : "",
    t.vertexUv3s ? "#define USE_UV3" : "",
    t.pointsUvs ? "#define USE_POINTS_UV" : "",
    t.flatShading ? "#define FLAT_SHADED" : "",
    t.skinning ? "#define USE_SKINNING" : "",
    t.morphTargets ? "#define USE_MORPHTARGETS" : "",
    t.morphNormals && t.flatShading === !1 ? "#define USE_MORPHNORMALS" : "",
    t.morphColors ? "#define USE_MORPHCOLORS" : "",
    t.morphTargetsCount > 0 ? "#define MORPHTARGETS_TEXTURE_STRIDE " + t.morphTextureStride : "",
    t.morphTargetsCount > 0 ? "#define MORPHTARGETS_COUNT " + t.morphTargetsCount : "",
    t.doubleSided ? "#define DOUBLE_SIDED" : "",
    t.flipSided ? "#define FLIP_SIDED" : "",
    t.shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
    t.shadowMapEnabled ? "#define " + l : "",
    t.sizeAttenuation ? "#define USE_SIZEATTENUATION" : "",
    t.numLightProbes > 0 ? "#define USE_LIGHT_PROBES" : "",
    t.logarithmicDepthBuffer ? "#define USE_LOGARITHMIC_DEPTH_BUFFER" : "",
    t.reversedDepthBuffer ? "#define USE_REVERSED_DEPTH_BUFFER" : "",
    "uniform mat4 modelMatrix;",
    "uniform mat4 modelViewMatrix;",
    "uniform mat4 projectionMatrix;",
    "uniform mat4 viewMatrix;",
    "uniform mat3 normalMatrix;",
    "uniform vec3 cameraPosition;",
    "uniform bool isOrthographic;",
    "#ifdef USE_INSTANCING",
    "	attribute mat4 instanceMatrix;",
    "#endif",
    "#ifdef USE_INSTANCING_COLOR",
    "	attribute vec3 instanceColor;",
    "#endif",
    "#ifdef USE_INSTANCING_MORPH",
    "	uniform sampler2D morphTexture;",
    "#endif",
    "attribute vec3 position;",
    "attribute vec3 normal;",
    "attribute vec2 uv;",
    "#ifdef USE_UV1",
    "	attribute vec2 uv1;",
    "#endif",
    "#ifdef USE_UV2",
    "	attribute vec2 uv2;",
    "#endif",
    "#ifdef USE_UV3",
    "	attribute vec2 uv3;",
    "#endif",
    "#ifdef USE_TANGENT",
    "	attribute vec4 tangent;",
    "#endif",
    "#if defined( USE_COLOR_ALPHA )",
    "	attribute vec4 color;",
    "#elif defined( USE_COLOR )",
    "	attribute vec3 color;",
    "#endif",
    "#ifdef USE_SKINNING",
    "	attribute vec4 skinIndex;",
    "	attribute vec4 skinWeight;",
    "#endif",
    `
`
  ].filter(or).join(`
`), d = [
    wh(t),
    "#define SHADER_TYPE " + t.shaderType,
    "#define SHADER_NAME " + t.shaderName,
    v,
    t.useFog && t.fog ? "#define USE_FOG" : "",
    t.useFog && t.fogExp2 ? "#define FOG_EXP2" : "",
    t.alphaToCoverage ? "#define ALPHA_TO_COVERAGE" : "",
    t.map ? "#define USE_MAP" : "",
    t.matcap ? "#define USE_MATCAP" : "",
    t.envMap ? "#define USE_ENVMAP" : "",
    t.envMap ? "#define " + c : "",
    t.envMap ? "#define " + u : "",
    t.envMap ? "#define " + h : "",
    f ? "#define CUBEUV_TEXEL_WIDTH " + f.texelWidth : "",
    f ? "#define CUBEUV_TEXEL_HEIGHT " + f.texelHeight : "",
    f ? "#define CUBEUV_MAX_MIP " + f.maxMip + ".0" : "",
    t.lightMap ? "#define USE_LIGHTMAP" : "",
    t.aoMap ? "#define USE_AOMAP" : "",
    t.bumpMap ? "#define USE_BUMPMAP" : "",
    t.normalMap ? "#define USE_NORMALMAP" : "",
    t.normalMapObjectSpace ? "#define USE_NORMALMAP_OBJECTSPACE" : "",
    t.normalMapTangentSpace ? "#define USE_NORMALMAP_TANGENTSPACE" : "",
    t.emissiveMap ? "#define USE_EMISSIVEMAP" : "",
    t.anisotropy ? "#define USE_ANISOTROPY" : "",
    t.anisotropyMap ? "#define USE_ANISOTROPYMAP" : "",
    t.clearcoat ? "#define USE_CLEARCOAT" : "",
    t.clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
    t.clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
    t.clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
    t.dispersion ? "#define USE_DISPERSION" : "",
    t.iridescence ? "#define USE_IRIDESCENCE" : "",
    t.iridescenceMap ? "#define USE_IRIDESCENCEMAP" : "",
    t.iridescenceThicknessMap ? "#define USE_IRIDESCENCE_THICKNESSMAP" : "",
    t.specularMap ? "#define USE_SPECULARMAP" : "",
    t.specularColorMap ? "#define USE_SPECULAR_COLORMAP" : "",
    t.specularIntensityMap ? "#define USE_SPECULAR_INTENSITYMAP" : "",
    t.roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
    t.metalnessMap ? "#define USE_METALNESSMAP" : "",
    t.alphaMap ? "#define USE_ALPHAMAP" : "",
    t.alphaTest ? "#define USE_ALPHATEST" : "",
    t.alphaHash ? "#define USE_ALPHAHASH" : "",
    t.sheen ? "#define USE_SHEEN" : "",
    t.sheenColorMap ? "#define USE_SHEEN_COLORMAP" : "",
    t.sheenRoughnessMap ? "#define USE_SHEEN_ROUGHNESSMAP" : "",
    t.transmission ? "#define USE_TRANSMISSION" : "",
    t.transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
    t.thicknessMap ? "#define USE_THICKNESSMAP" : "",
    t.vertexTangents && t.flatShading === !1 ? "#define USE_TANGENT" : "",
    t.vertexColors || t.instancingColor || t.batchingColor ? "#define USE_COLOR" : "",
    t.vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
    t.vertexUv1s ? "#define USE_UV1" : "",
    t.vertexUv2s ? "#define USE_UV2" : "",
    t.vertexUv3s ? "#define USE_UV3" : "",
    t.pointsUvs ? "#define USE_POINTS_UV" : "",
    t.gradientMap ? "#define USE_GRADIENTMAP" : "",
    t.flatShading ? "#define FLAT_SHADED" : "",
    t.doubleSided ? "#define DOUBLE_SIDED" : "",
    t.flipSided ? "#define FLIP_SIDED" : "",
    t.shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
    t.shadowMapEnabled ? "#define " + l : "",
    t.premultipliedAlpha ? "#define PREMULTIPLIED_ALPHA" : "",
    t.numLightProbes > 0 ? "#define USE_LIGHT_PROBES" : "",
    t.decodeVideoTexture ? "#define DECODE_VIDEO_TEXTURE" : "",
    t.decodeVideoTextureEmissive ? "#define DECODE_VIDEO_TEXTURE_EMISSIVE" : "",
    t.logarithmicDepthBuffer ? "#define USE_LOGARITHMIC_DEPTH_BUFFER" : "",
    t.reversedDepthBuffer ? "#define USE_REVERSED_DEPTH_BUFFER" : "",
    "uniform mat4 viewMatrix;",
    "uniform vec3 cameraPosition;",
    "uniform bool isOrthographic;",
    t.toneMapping !== xi ? "#define TONE_MAPPING" : "",
    t.toneMapping !== xi ? qe.tonemapping_pars_fragment : "",
    // this code is required here because it is used by the toneMapping() function defined below
    t.toneMapping !== xi ? hS("toneMapping", t.toneMapping) : "",
    t.dithering ? "#define DITHERING" : "",
    t.opaque ? "#define OPAQUE" : "",
    qe.colorspace_pars_fragment,
    // this code is required here because it is used by the various encoding/decoding function defined below
    uS("linearToOutputTexel", t.outputColorSpace),
    fS(),
    t.useDepthPacking ? "#define DEPTH_PACKING " + t.depthPacking : "",
    `
`
  ].filter(or).join(`
`)), o = tc(o), o = Th(o, t), o = bh(o, t), a = tc(a), a = Th(a, t), a = bh(a, t), o = Ah(o), a = Ah(a), t.isRawShaderMaterial !== !0 && (b = `#version 300 es
`, m = [
    p,
    "#define attribute in",
    "#define varying out",
    "#define texture2D texture"
  ].join(`
`) + `
` + m, d = [
    "#define varying in",
    t.glslVersion === Pu ? "" : "layout(location = 0) out highp vec4 pc_fragColor;",
    t.glslVersion === Pu ? "" : "#define gl_FragColor pc_fragColor",
    "#define gl_FragDepthEXT gl_FragDepth",
    "#define texture2D texture",
    "#define textureCube texture",
    "#define texture2DProj textureProj",
    "#define texture2DLodEXT textureLod",
    "#define texture2DProjLodEXT textureProjLod",
    "#define textureCubeLodEXT textureLod",
    "#define texture2DGradEXT textureGrad",
    "#define texture2DProjGradEXT textureProjGrad",
    "#define textureCubeGradEXT textureGrad"
  ].join(`
`) + `
` + d);
  const A = b + m + o, M = b + d + a, R = Sh(s, s.VERTEX_SHADER, A), w = Sh(s, s.FRAGMENT_SHADER, M);
  s.attachShader(x, R), s.attachShader(x, w), t.index0AttributeName !== void 0 ? s.bindAttribLocation(x, 0, t.index0AttributeName) : t.morphTargets === !0 && s.bindAttribLocation(x, 0, "position"), s.linkProgram(x);
  function D(P) {
    if (n.debug.checkShaderErrors) {
      const L = s.getProgramInfoLog(x) || "", V = s.getShaderInfoLog(R) || "", Z = s.getShaderInfoLog(w) || "", te = L.trim(), $ = V.trim(), ie = Z.trim();
      let H = !0, fe = !0;
      if (s.getProgramParameter(x, s.LINK_STATUS) === !1)
        if (H = !1, typeof n.debug.onShaderError == "function")
          n.debug.onShaderError(s, x, R, w);
        else {
          const xe = Eh(s, R, "vertex"), me = Eh(s, w, "fragment");
          console.error(
            "THREE.WebGLProgram: Shader Error " + s.getError() + " - VALIDATE_STATUS " + s.getProgramParameter(x, s.VALIDATE_STATUS) + `

Material Name: ` + P.name + `
Material Type: ` + P.type + `

Program Info Log: ` + te + `
` + xe + `
` + me
          );
        }
      else te !== "" ? console.warn("THREE.WebGLProgram: Program Info Log:", te) : ($ === "" || ie === "") && (fe = !1);
      fe && (P.diagnostics = {
        runnable: H,
        programLog: te,
        vertexShader: {
          log: $,
          prefix: m
        },
        fragmentShader: {
          log: ie,
          prefix: d
        }
      });
    }
    s.deleteShader(R), s.deleteShader(w), U = new Ao(s, x), y = mS(s, x);
  }
  let U;
  this.getUniforms = function() {
    return U === void 0 && D(this), U;
  };
  let y;
  this.getAttributes = function() {
    return y === void 0 && D(this), y;
  };
  let S = t.rendererExtensionParallelShaderCompile === !1;
  return this.isReady = function() {
    return S === !1 && (S = s.getProgramParameter(x, oS)), S;
  }, this.destroy = function() {
    i.releaseStatesOfProgram(this), s.deleteProgram(x), this.program = void 0;
  }, this.type = t.shaderType, this.name = t.shaderName, this.id = aS++, this.cacheKey = e, this.usedTimes = 1, this.program = x, this.vertexShader = R, this.fragmentShader = w, this;
}
let wS = 0;
class RS {
  constructor() {
    this.shaderCache = /* @__PURE__ */ new Map(), this.materialCache = /* @__PURE__ */ new Map();
  }
  update(e) {
    const t = e.vertexShader, i = e.fragmentShader, s = this._getShaderStage(t), r = this._getShaderStage(i), o = this._getShaderCacheForMaterial(e);
    return o.has(s) === !1 && (o.add(s), s.usedTimes++), o.has(r) === !1 && (o.add(r), r.usedTimes++), this;
  }
  remove(e) {
    const t = this.materialCache.get(e);
    for (const i of t)
      i.usedTimes--, i.usedTimes === 0 && this.shaderCache.delete(i.code);
    return this.materialCache.delete(e), this;
  }
  getVertexShaderID(e) {
    return this._getShaderStage(e.vertexShader).id;
  }
  getFragmentShaderID(e) {
    return this._getShaderStage(e.fragmentShader).id;
  }
  dispose() {
    this.shaderCache.clear(), this.materialCache.clear();
  }
  _getShaderCacheForMaterial(e) {
    const t = this.materialCache;
    let i = t.get(e);
    return i === void 0 && (i = /* @__PURE__ */ new Set(), t.set(e, i)), i;
  }
  _getShaderStage(e) {
    const t = this.shaderCache;
    let i = t.get(e);
    return i === void 0 && (i = new CS(e), t.set(e, i)), i;
  }
}
class CS {
  constructor(e) {
    this.id = wS++, this.code = e, this.usedTimes = 0;
  }
}
function PS(n, e, t, i, s, r, o) {
  const a = new pd(), l = new RS(), c = /* @__PURE__ */ new Set(), u = [], h = s.logarithmicDepthBuffer, f = s.vertexTextures;
  let p = s.precision;
  const v = {
    MeshDepthMaterial: "depth",
    MeshDistanceMaterial: "distanceRGBA",
    MeshNormalMaterial: "normal",
    MeshBasicMaterial: "basic",
    MeshLambertMaterial: "lambert",
    MeshPhongMaterial: "phong",
    MeshToonMaterial: "toon",
    MeshStandardMaterial: "physical",
    MeshPhysicalMaterial: "physical",
    MeshMatcapMaterial: "matcap",
    LineBasicMaterial: "basic",
    LineDashedMaterial: "dashed",
    PointsMaterial: "points",
    ShadowMaterial: "shadow",
    SpriteMaterial: "sprite"
  };
  function x(y) {
    return c.add(y), y === 0 ? "uv" : `uv${y}`;
  }
  function m(y, S, P, L, V) {
    const Z = L.fog, te = V.geometry, $ = y.isMeshStandardMaterial ? L.environment : null, ie = (y.isMeshStandardMaterial ? t : e).get(y.envMap || $), H = ie && ie.mapping === ta ? ie.image.height : null, fe = v[y.type];
    y.precision !== null && (p = s.getMaxPrecision(y.precision), p !== y.precision && console.warn("THREE.WebGLProgram.getParameters:", y.precision, "not supported, using", p, "instead."));
    const xe = te.morphAttributes.position || te.morphAttributes.normal || te.morphAttributes.color, me = xe !== void 0 ? xe.length : 0;
    let de = 0;
    te.morphAttributes.position !== void 0 && (de = 1), te.morphAttributes.normal !== void 0 && (de = 2), te.morphAttributes.color !== void 0 && (de = 3);
    let Le, tt, Ze, ne;
    if (fe) {
      const nt = Ln[fe];
      Le = nt.vertexShader, tt = nt.fragmentShader;
    } else
      Le = y.vertexShader, tt = y.fragmentShader, l.update(y), Ze = l.getVertexShaderID(y), ne = l.getFragmentShaderID(y);
    const re = n.getRenderTarget(), Ae = n.state.buffers.depth.getReversed(), Oe = V.isInstancedMesh === !0, Pe = V.isBatchedMesh === !0, $e = !!y.map, C = !!y.matcap, g = !!ie, W = !!y.aoMap, j = !!y.lightMap, X = !!y.bumpMap, z = !!y.normalMap, ae = !!y.displacementMap, q = !!y.emissiveMap, Q = !!y.metalnessMap, ee = !!y.roughnessMap, Se = y.anisotropy > 0, E = y.clearcoat > 0, _ = y.dispersion > 0, I = y.iridescence > 0, k = y.sheen > 0, J = y.transmission > 0, G = Se && !!y.anisotropyMap, _e = E && !!y.clearcoatMap, oe = E && !!y.clearcoatNormalMap, Ee = E && !!y.clearcoatRoughnessMap, Te = I && !!y.iridescenceMap, le = I && !!y.iridescenceThicknessMap, Me = k && !!y.sheenColorMap, Ce = k && !!y.sheenRoughnessMap, be = !!y.specularMap, ge = !!y.specularColorMap, ke = !!y.specularIntensityMap, F = J && !!y.transmissionMap, he = J && !!y.thicknessMap, pe = !!y.gradientMap, Re = !!y.alphaMap, ce = y.alphaTest > 0, se = !!y.alphaHash, Ie = !!y.extensions;
    let Ge = xi;
    y.toneMapped && (re === null || re.isXRRenderTarget === !0) && (Ge = n.toneMapping);
    const ht = {
      shaderID: fe,
      shaderType: y.type,
      shaderName: y.name,
      vertexShader: Le,
      fragmentShader: tt,
      defines: y.defines,
      customVertexShaderID: Ze,
      customFragmentShaderID: ne,
      isRawShaderMaterial: y.isRawShaderMaterial === !0,
      glslVersion: y.glslVersion,
      precision: p,
      batching: Pe,
      batchingColor: Pe && V._colorsTexture !== null,
      instancing: Oe,
      instancingColor: Oe && V.instanceColor !== null,
      instancingMorph: Oe && V.morphTexture !== null,
      supportsVertexTextures: f,
      outputColorSpace: re === null ? n.outputColorSpace : re.isXRRenderTarget === !0 ? re.texture.colorSpace : Os,
      alphaToCoverage: !!y.alphaToCoverage,
      map: $e,
      matcap: C,
      envMap: g,
      envMapMode: g && ie.mapping,
      envMapCubeUVHeight: H,
      aoMap: W,
      lightMap: j,
      bumpMap: X,
      normalMap: z,
      displacementMap: f && ae,
      emissiveMap: q,
      normalMapObjectSpace: z && y.normalMapType === xg,
      normalMapTangentSpace: z && y.normalMapType === ud,
      metalnessMap: Q,
      roughnessMap: ee,
      anisotropy: Se,
      anisotropyMap: G,
      clearcoat: E,
      clearcoatMap: _e,
      clearcoatNormalMap: oe,
      clearcoatRoughnessMap: Ee,
      dispersion: _,
      iridescence: I,
      iridescenceMap: Te,
      iridescenceThicknessMap: le,
      sheen: k,
      sheenColorMap: Me,
      sheenRoughnessMap: Ce,
      specularMap: be,
      specularColorMap: ge,
      specularIntensityMap: ke,
      transmission: J,
      transmissionMap: F,
      thicknessMap: he,
      gradientMap: pe,
      opaque: y.transparent === !1 && y.blending === Ds && y.alphaToCoverage === !1,
      alphaMap: Re,
      alphaTest: ce,
      alphaHash: se,
      combine: y.combine,
      //
      mapUv: $e && x(y.map.channel),
      aoMapUv: W && x(y.aoMap.channel),
      lightMapUv: j && x(y.lightMap.channel),
      bumpMapUv: X && x(y.bumpMap.channel),
      normalMapUv: z && x(y.normalMap.channel),
      displacementMapUv: ae && x(y.displacementMap.channel),
      emissiveMapUv: q && x(y.emissiveMap.channel),
      metalnessMapUv: Q && x(y.metalnessMap.channel),
      roughnessMapUv: ee && x(y.roughnessMap.channel),
      anisotropyMapUv: G && x(y.anisotropyMap.channel),
      clearcoatMapUv: _e && x(y.clearcoatMap.channel),
      clearcoatNormalMapUv: oe && x(y.clearcoatNormalMap.channel),
      clearcoatRoughnessMapUv: Ee && x(y.clearcoatRoughnessMap.channel),
      iridescenceMapUv: Te && x(y.iridescenceMap.channel),
      iridescenceThicknessMapUv: le && x(y.iridescenceThicknessMap.channel),
      sheenColorMapUv: Me && x(y.sheenColorMap.channel),
      sheenRoughnessMapUv: Ce && x(y.sheenRoughnessMap.channel),
      specularMapUv: be && x(y.specularMap.channel),
      specularColorMapUv: ge && x(y.specularColorMap.channel),
      specularIntensityMapUv: ke && x(y.specularIntensityMap.channel),
      transmissionMapUv: F && x(y.transmissionMap.channel),
      thicknessMapUv: he && x(y.thicknessMap.channel),
      alphaMapUv: Re && x(y.alphaMap.channel),
      //
      vertexTangents: !!te.attributes.tangent && (z || Se),
      vertexColors: y.vertexColors,
      vertexAlphas: y.vertexColors === !0 && !!te.attributes.color && te.attributes.color.itemSize === 4,
      pointsUvs: V.isPoints === !0 && !!te.attributes.uv && ($e || Re),
      fog: !!Z,
      useFog: y.fog === !0,
      fogExp2: !!Z && Z.isFogExp2,
      flatShading: y.flatShading === !0 && y.wireframe === !1,
      sizeAttenuation: y.sizeAttenuation === !0,
      logarithmicDepthBuffer: h,
      reversedDepthBuffer: Ae,
      skinning: V.isSkinnedMesh === !0,
      morphTargets: te.morphAttributes.position !== void 0,
      morphNormals: te.morphAttributes.normal !== void 0,
      morphColors: te.morphAttributes.color !== void 0,
      morphTargetsCount: me,
      morphTextureStride: de,
      numDirLights: S.directional.length,
      numPointLights: S.point.length,
      numSpotLights: S.spot.length,
      numSpotLightMaps: S.spotLightMap.length,
      numRectAreaLights: S.rectArea.length,
      numHemiLights: S.hemi.length,
      numDirLightShadows: S.directionalShadowMap.length,
      numPointLightShadows: S.pointShadowMap.length,
      numSpotLightShadows: S.spotShadowMap.length,
      numSpotLightShadowsWithMaps: S.numSpotLightShadowsWithMaps,
      numLightProbes: S.numLightProbes,
      numClippingPlanes: o.numPlanes,
      numClipIntersection: o.numIntersection,
      dithering: y.dithering,
      shadowMapEnabled: n.shadowMap.enabled && P.length > 0,
      shadowMapType: n.shadowMap.type,
      toneMapping: Ge,
      decodeVideoTexture: $e && y.map.isVideoTexture === !0 && Qe.getTransfer(y.map.colorSpace) === ot,
      decodeVideoTextureEmissive: q && y.emissiveMap.isVideoTexture === !0 && Qe.getTransfer(y.emissiveMap.colorSpace) === ot,
      premultipliedAlpha: y.premultipliedAlpha,
      doubleSided: y.side === Qn,
      flipSided: y.side === Wt,
      useDepthPacking: y.depthPacking >= 0,
      depthPacking: y.depthPacking || 0,
      index0AttributeName: y.index0AttributeName,
      extensionClipCullDistance: Ie && y.extensions.clipCullDistance === !0 && i.has("WEBGL_clip_cull_distance"),
      extensionMultiDraw: (Ie && y.extensions.multiDraw === !0 || Pe) && i.has("WEBGL_multi_draw"),
      rendererExtensionParallelShaderCompile: i.has("KHR_parallel_shader_compile"),
      customProgramCacheKey: y.customProgramCacheKey()
    };
    return ht.vertexUv1s = c.has(1), ht.vertexUv2s = c.has(2), ht.vertexUv3s = c.has(3), c.clear(), ht;
  }
  function d(y) {
    const S = [];
    if (y.shaderID ? S.push(y.shaderID) : (S.push(y.customVertexShaderID), S.push(y.customFragmentShaderID)), y.defines !== void 0)
      for (const P in y.defines)
        S.push(P), S.push(y.defines[P]);
    return y.isRawShaderMaterial === !1 && (b(S, y), A(S, y), S.push(n.outputColorSpace)), S.push(y.customProgramCacheKey), S.join();
  }
  function b(y, S) {
    y.push(S.precision), y.push(S.outputColorSpace), y.push(S.envMapMode), y.push(S.envMapCubeUVHeight), y.push(S.mapUv), y.push(S.alphaMapUv), y.push(S.lightMapUv), y.push(S.aoMapUv), y.push(S.bumpMapUv), y.push(S.normalMapUv), y.push(S.displacementMapUv), y.push(S.emissiveMapUv), y.push(S.metalnessMapUv), y.push(S.roughnessMapUv), y.push(S.anisotropyMapUv), y.push(S.clearcoatMapUv), y.push(S.clearcoatNormalMapUv), y.push(S.clearcoatRoughnessMapUv), y.push(S.iridescenceMapUv), y.push(S.iridescenceThicknessMapUv), y.push(S.sheenColorMapUv), y.push(S.sheenRoughnessMapUv), y.push(S.specularMapUv), y.push(S.specularColorMapUv), y.push(S.specularIntensityMapUv), y.push(S.transmissionMapUv), y.push(S.thicknessMapUv), y.push(S.combine), y.push(S.fogExp2), y.push(S.sizeAttenuation), y.push(S.morphTargetsCount), y.push(S.morphAttributeCount), y.push(S.numDirLights), y.push(S.numPointLights), y.push(S.numSpotLights), y.push(S.numSpotLightMaps), y.push(S.numHemiLights), y.push(S.numRectAreaLights), y.push(S.numDirLightShadows), y.push(S.numPointLightShadows), y.push(S.numSpotLightShadows), y.push(S.numSpotLightShadowsWithMaps), y.push(S.numLightProbes), y.push(S.shadowMapType), y.push(S.toneMapping), y.push(S.numClippingPlanes), y.push(S.numClipIntersection), y.push(S.depthPacking);
  }
  function A(y, S) {
    a.disableAll(), S.supportsVertexTextures && a.enable(0), S.instancing && a.enable(1), S.instancingColor && a.enable(2), S.instancingMorph && a.enable(3), S.matcap && a.enable(4), S.envMap && a.enable(5), S.normalMapObjectSpace && a.enable(6), S.normalMapTangentSpace && a.enable(7), S.clearcoat && a.enable(8), S.iridescence && a.enable(9), S.alphaTest && a.enable(10), S.vertexColors && a.enable(11), S.vertexAlphas && a.enable(12), S.vertexUv1s && a.enable(13), S.vertexUv2s && a.enable(14), S.vertexUv3s && a.enable(15), S.vertexTangents && a.enable(16), S.anisotropy && a.enable(17), S.alphaHash && a.enable(18), S.batching && a.enable(19), S.dispersion && a.enable(20), S.batchingColor && a.enable(21), S.gradientMap && a.enable(22), y.push(a.mask), a.disableAll(), S.fog && a.enable(0), S.useFog && a.enable(1), S.flatShading && a.enable(2), S.logarithmicDepthBuffer && a.enable(3), S.reversedDepthBuffer && a.enable(4), S.skinning && a.enable(5), S.morphTargets && a.enable(6), S.morphNormals && a.enable(7), S.morphColors && a.enable(8), S.premultipliedAlpha && a.enable(9), S.shadowMapEnabled && a.enable(10), S.doubleSided && a.enable(11), S.flipSided && a.enable(12), S.useDepthPacking && a.enable(13), S.dithering && a.enable(14), S.transmission && a.enable(15), S.sheen && a.enable(16), S.opaque && a.enable(17), S.pointsUvs && a.enable(18), S.decodeVideoTexture && a.enable(19), S.decodeVideoTextureEmissive && a.enable(20), S.alphaToCoverage && a.enable(21), y.push(a.mask);
  }
  function M(y) {
    const S = v[y.type];
    let P;
    if (S) {
      const L = Ln[S];
      P = Kg.clone(L.uniforms);
    } else
      P = y.uniforms;
    return P;
  }
  function R(y, S) {
    let P;
    for (let L = 0, V = u.length; L < V; L++) {
      const Z = u[L];
      if (Z.cacheKey === S) {
        P = Z, ++P.usedTimes;
        break;
      }
    }
    return P === void 0 && (P = new AS(n, S, y, r), u.push(P)), P;
  }
  function w(y) {
    if (--y.usedTimes === 0) {
      const S = u.indexOf(y);
      u[S] = u[u.length - 1], u.pop(), y.destroy();
    }
  }
  function D(y) {
    l.remove(y);
  }
  function U() {
    l.dispose();
  }
  return {
    getParameters: m,
    getProgramCacheKey: d,
    getUniforms: M,
    acquireProgram: R,
    releaseProgram: w,
    releaseShaderCache: D,
    // Exposed for resource monitoring & error feedback via renderer.info:
    programs: u,
    dispose: U
  };
}
function DS() {
  let n = /* @__PURE__ */ new WeakMap();
  function e(o) {
    return n.has(o);
  }
  function t(o) {
    let a = n.get(o);
    return a === void 0 && (a = {}, n.set(o, a)), a;
  }
  function i(o) {
    n.delete(o);
  }
  function s(o, a, l) {
    n.get(o)[a] = l;
  }
  function r() {
    n = /* @__PURE__ */ new WeakMap();
  }
  return {
    has: e,
    get: t,
    remove: i,
    update: s,
    dispose: r
  };
}
function LS(n, e) {
  return n.groupOrder !== e.groupOrder ? n.groupOrder - e.groupOrder : n.renderOrder !== e.renderOrder ? n.renderOrder - e.renderOrder : n.material.id !== e.material.id ? n.material.id - e.material.id : n.z !== e.z ? n.z - e.z : n.id - e.id;
}
function Rh(n, e) {
  return n.groupOrder !== e.groupOrder ? n.groupOrder - e.groupOrder : n.renderOrder !== e.renderOrder ? n.renderOrder - e.renderOrder : n.z !== e.z ? e.z - n.z : n.id - e.id;
}
function Ch() {
  const n = [];
  let e = 0;
  const t = [], i = [], s = [];
  function r() {
    e = 0, t.length = 0, i.length = 0, s.length = 0;
  }
  function o(h, f, p, v, x, m) {
    let d = n[e];
    return d === void 0 ? (d = {
      id: h.id,
      object: h,
      geometry: f,
      material: p,
      groupOrder: v,
      renderOrder: h.renderOrder,
      z: x,
      group: m
    }, n[e] = d) : (d.id = h.id, d.object = h, d.geometry = f, d.material = p, d.groupOrder = v, d.renderOrder = h.renderOrder, d.z = x, d.group = m), e++, d;
  }
  function a(h, f, p, v, x, m) {
    const d = o(h, f, p, v, x, m);
    p.transmission > 0 ? i.push(d) : p.transparent === !0 ? s.push(d) : t.push(d);
  }
  function l(h, f, p, v, x, m) {
    const d = o(h, f, p, v, x, m);
    p.transmission > 0 ? i.unshift(d) : p.transparent === !0 ? s.unshift(d) : t.unshift(d);
  }
  function c(h, f) {
    t.length > 1 && t.sort(h || LS), i.length > 1 && i.sort(f || Rh), s.length > 1 && s.sort(f || Rh);
  }
  function u() {
    for (let h = e, f = n.length; h < f; h++) {
      const p = n[h];
      if (p.id === null) break;
      p.id = null, p.object = null, p.geometry = null, p.material = null, p.group = null;
    }
  }
  return {
    opaque: t,
    transmissive: i,
    transparent: s,
    init: r,
    push: a,
    unshift: l,
    finish: u,
    sort: c
  };
}
function IS() {
  let n = /* @__PURE__ */ new WeakMap();
  function e(i, s) {
    const r = n.get(i);
    let o;
    return r === void 0 ? (o = new Ch(), n.set(i, [o])) : s >= r.length ? (o = new Ch(), r.push(o)) : o = r[s], o;
  }
  function t() {
    n = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: e,
    dispose: t
  };
}
function US() {
  const n = {};
  return {
    get: function(e) {
      if (n[e.id] !== void 0)
        return n[e.id];
      let t;
      switch (e.type) {
        case "DirectionalLight":
          t = {
            direction: new N(),
            color: new We()
          };
          break;
        case "SpotLight":
          t = {
            position: new N(),
            direction: new N(),
            color: new We(),
            distance: 0,
            coneCos: 0,
            penumbraCos: 0,
            decay: 0
          };
          break;
        case "PointLight":
          t = {
            position: new N(),
            color: new We(),
            distance: 0,
            decay: 0
          };
          break;
        case "HemisphereLight":
          t = {
            direction: new N(),
            skyColor: new We(),
            groundColor: new We()
          };
          break;
        case "RectAreaLight":
          t = {
            color: new We(),
            position: new N(),
            halfWidth: new N(),
            halfHeight: new N()
          };
          break;
      }
      return n[e.id] = t, t;
    }
  };
}
function NS() {
  const n = {};
  return {
    get: function(e) {
      if (n[e.id] !== void 0)
        return n[e.id];
      let t;
      switch (e.type) {
        case "DirectionalLight":
          t = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new Ve()
          };
          break;
        case "SpotLight":
          t = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new Ve()
          };
          break;
        case "PointLight":
          t = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new Ve(),
            shadowCameraNear: 1,
            shadowCameraFar: 1e3
          };
          break;
      }
      return n[e.id] = t, t;
    }
  };
}
let FS = 0;
function OS(n, e) {
  return (e.castShadow ? 2 : 0) - (n.castShadow ? 2 : 0) + (e.map ? 1 : 0) - (n.map ? 1 : 0);
}
function BS(n) {
  const e = new US(), t = NS(), i = {
    version: 0,
    hash: {
      directionalLength: -1,
      pointLength: -1,
      spotLength: -1,
      rectAreaLength: -1,
      hemiLength: -1,
      numDirectionalShadows: -1,
      numPointShadows: -1,
      numSpotShadows: -1,
      numSpotMaps: -1,
      numLightProbes: -1
    },
    ambient: [0, 0, 0],
    probe: [],
    directional: [],
    directionalShadow: [],
    directionalShadowMap: [],
    directionalShadowMatrix: [],
    spot: [],
    spotLightMap: [],
    spotShadow: [],
    spotShadowMap: [],
    spotLightMatrix: [],
    rectArea: [],
    rectAreaLTC1: null,
    rectAreaLTC2: null,
    point: [],
    pointShadow: [],
    pointShadowMap: [],
    pointShadowMatrix: [],
    hemi: [],
    numSpotLightShadowsWithMaps: 0,
    numLightProbes: 0
  };
  for (let c = 0; c < 9; c++) i.probe.push(new N());
  const s = new N(), r = new pt(), o = new pt();
  function a(c) {
    let u = 0, h = 0, f = 0;
    for (let y = 0; y < 9; y++) i.probe[y].set(0, 0, 0);
    let p = 0, v = 0, x = 0, m = 0, d = 0, b = 0, A = 0, M = 0, R = 0, w = 0, D = 0;
    c.sort(OS);
    for (let y = 0, S = c.length; y < S; y++) {
      const P = c[y], L = P.color, V = P.intensity, Z = P.distance, te = P.shadow && P.shadow.map ? P.shadow.map.texture : null;
      if (P.isAmbientLight)
        u += L.r * V, h += L.g * V, f += L.b * V;
      else if (P.isLightProbe) {
        for (let $ = 0; $ < 9; $++)
          i.probe[$].addScaledVector(P.sh.coefficients[$], V);
        D++;
      } else if (P.isDirectionalLight) {
        const $ = e.get(P);
        if ($.color.copy(P.color).multiplyScalar(P.intensity), P.castShadow) {
          const ie = P.shadow, H = t.get(P);
          H.shadowIntensity = ie.intensity, H.shadowBias = ie.bias, H.shadowNormalBias = ie.normalBias, H.shadowRadius = ie.radius, H.shadowMapSize = ie.mapSize, i.directionalShadow[p] = H, i.directionalShadowMap[p] = te, i.directionalShadowMatrix[p] = P.shadow.matrix, b++;
        }
        i.directional[p] = $, p++;
      } else if (P.isSpotLight) {
        const $ = e.get(P);
        $.position.setFromMatrixPosition(P.matrixWorld), $.color.copy(L).multiplyScalar(V), $.distance = Z, $.coneCos = Math.cos(P.angle), $.penumbraCos = Math.cos(P.angle * (1 - P.penumbra)), $.decay = P.decay, i.spot[x] = $;
        const ie = P.shadow;
        if (P.map && (i.spotLightMap[R] = P.map, R++, ie.updateMatrices(P), P.castShadow && w++), i.spotLightMatrix[x] = ie.matrix, P.castShadow) {
          const H = t.get(P);
          H.shadowIntensity = ie.intensity, H.shadowBias = ie.bias, H.shadowNormalBias = ie.normalBias, H.shadowRadius = ie.radius, H.shadowMapSize = ie.mapSize, i.spotShadow[x] = H, i.spotShadowMap[x] = te, M++;
        }
        x++;
      } else if (P.isRectAreaLight) {
        const $ = e.get(P);
        $.color.copy(L).multiplyScalar(V), $.halfWidth.set(P.width * 0.5, 0, 0), $.halfHeight.set(0, P.height * 0.5, 0), i.rectArea[m] = $, m++;
      } else if (P.isPointLight) {
        const $ = e.get(P);
        if ($.color.copy(P.color).multiplyScalar(P.intensity), $.distance = P.distance, $.decay = P.decay, P.castShadow) {
          const ie = P.shadow, H = t.get(P);
          H.shadowIntensity = ie.intensity, H.shadowBias = ie.bias, H.shadowNormalBias = ie.normalBias, H.shadowRadius = ie.radius, H.shadowMapSize = ie.mapSize, H.shadowCameraNear = ie.camera.near, H.shadowCameraFar = ie.camera.far, i.pointShadow[v] = H, i.pointShadowMap[v] = te, i.pointShadowMatrix[v] = P.shadow.matrix, A++;
        }
        i.point[v] = $, v++;
      } else if (P.isHemisphereLight) {
        const $ = e.get(P);
        $.skyColor.copy(P.color).multiplyScalar(V), $.groundColor.copy(P.groundColor).multiplyScalar(V), i.hemi[d] = $, d++;
      }
    }
    m > 0 && (n.has("OES_texture_float_linear") === !0 ? (i.rectAreaLTC1 = ve.LTC_FLOAT_1, i.rectAreaLTC2 = ve.LTC_FLOAT_2) : (i.rectAreaLTC1 = ve.LTC_HALF_1, i.rectAreaLTC2 = ve.LTC_HALF_2)), i.ambient[0] = u, i.ambient[1] = h, i.ambient[2] = f;
    const U = i.hash;
    (U.directionalLength !== p || U.pointLength !== v || U.spotLength !== x || U.rectAreaLength !== m || U.hemiLength !== d || U.numDirectionalShadows !== b || U.numPointShadows !== A || U.numSpotShadows !== M || U.numSpotMaps !== R || U.numLightProbes !== D) && (i.directional.length = p, i.spot.length = x, i.rectArea.length = m, i.point.length = v, i.hemi.length = d, i.directionalShadow.length = b, i.directionalShadowMap.length = b, i.pointShadow.length = A, i.pointShadowMap.length = A, i.spotShadow.length = M, i.spotShadowMap.length = M, i.directionalShadowMatrix.length = b, i.pointShadowMatrix.length = A, i.spotLightMatrix.length = M + R - w, i.spotLightMap.length = R, i.numSpotLightShadowsWithMaps = w, i.numLightProbes = D, U.directionalLength = p, U.pointLength = v, U.spotLength = x, U.rectAreaLength = m, U.hemiLength = d, U.numDirectionalShadows = b, U.numPointShadows = A, U.numSpotShadows = M, U.numSpotMaps = R, U.numLightProbes = D, i.version = FS++);
  }
  function l(c, u) {
    let h = 0, f = 0, p = 0, v = 0, x = 0;
    const m = u.matrixWorldInverse;
    for (let d = 0, b = c.length; d < b; d++) {
      const A = c[d];
      if (A.isDirectionalLight) {
        const M = i.directional[h];
        M.direction.setFromMatrixPosition(A.matrixWorld), s.setFromMatrixPosition(A.target.matrixWorld), M.direction.sub(s), M.direction.transformDirection(m), h++;
      } else if (A.isSpotLight) {
        const M = i.spot[p];
        M.position.setFromMatrixPosition(A.matrixWorld), M.position.applyMatrix4(m), M.direction.setFromMatrixPosition(A.matrixWorld), s.setFromMatrixPosition(A.target.matrixWorld), M.direction.sub(s), M.direction.transformDirection(m), p++;
      } else if (A.isRectAreaLight) {
        const M = i.rectArea[v];
        M.position.setFromMatrixPosition(A.matrixWorld), M.position.applyMatrix4(m), o.identity(), r.copy(A.matrixWorld), r.premultiply(m), o.extractRotation(r), M.halfWidth.set(A.width * 0.5, 0, 0), M.halfHeight.set(0, A.height * 0.5, 0), M.halfWidth.applyMatrix4(o), M.halfHeight.applyMatrix4(o), v++;
      } else if (A.isPointLight) {
        const M = i.point[f];
        M.position.setFromMatrixPosition(A.matrixWorld), M.position.applyMatrix4(m), f++;
      } else if (A.isHemisphereLight) {
        const M = i.hemi[x];
        M.direction.setFromMatrixPosition(A.matrixWorld), M.direction.transformDirection(m), x++;
      }
    }
  }
  return {
    setup: a,
    setupView: l,
    state: i
  };
}
function Ph(n) {
  const e = new BS(n), t = [], i = [];
  function s(u) {
    c.camera = u, t.length = 0, i.length = 0;
  }
  function r(u) {
    t.push(u);
  }
  function o(u) {
    i.push(u);
  }
  function a() {
    e.setup(t);
  }
  function l(u) {
    e.setupView(t, u);
  }
  const c = {
    lightsArray: t,
    shadowsArray: i,
    camera: null,
    lights: e,
    transmissionRenderTarget: {}
  };
  return {
    init: s,
    state: c,
    setupLights: a,
    setupLightsView: l,
    pushLight: r,
    pushShadow: o
  };
}
function zS(n) {
  let e = /* @__PURE__ */ new WeakMap();
  function t(s, r = 0) {
    const o = e.get(s);
    let a;
    return o === void 0 ? (a = new Ph(n), e.set(s, [a])) : r >= o.length ? (a = new Ph(n), o.push(a)) : a = o[r], a;
  }
  function i() {
    e = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: t,
    dispose: i
  };
}
const HS = `void main() {
	gl_Position = vec4( position, 1.0 );
}`, VS = `uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;
function kS(n, e, t) {
  let i = new Ac();
  const s = new Ve(), r = new Ve(), o = new lt(), a = new l0({ depthPacking: vg }), l = new c0(), c = {}, u = t.maxTextureSize, h = { [Si]: Wt, [Wt]: Si, [Qn]: Qn }, f = new yi({
    defines: {
      VSM_SAMPLES: 8
    },
    uniforms: {
      shadow_pass: { value: null },
      resolution: { value: new Ve() },
      radius: { value: 4 }
    },
    vertexShader: HS,
    fragmentShader: VS
  }), p = f.clone();
  p.defines.HORIZONTAL_PASS = 1;
  const v = new Nt();
  v.setAttribute(
    "position",
    new En(
      new Float32Array([-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5]),
      3
    )
  );
  const x = new vt(v, f), m = this;
  this.enabled = !1, this.autoUpdate = !0, this.needsUpdate = !1, this.type = Jf;
  let d = this.type;
  this.render = function(w, D, U) {
    if (m.enabled === !1 || m.autoUpdate === !1 && m.needsUpdate === !1 || w.length === 0) return;
    const y = n.getRenderTarget(), S = n.getActiveCubeFace(), P = n.getActiveMipmapLevel(), L = n.state;
    L.setBlending(vi), L.buffers.depth.getReversed() === !0 ? L.buffers.color.setClear(0, 0, 0, 0) : L.buffers.color.setClear(1, 1, 1, 1), L.buffers.depth.setTest(!0), L.setScissorTest(!1);
    const V = d !== jn && this.type === jn, Z = d === jn && this.type !== jn;
    for (let te = 0, $ = w.length; te < $; te++) {
      const ie = w[te], H = ie.shadow;
      if (H === void 0) {
        console.warn("THREE.WebGLShadowMap:", ie, "has no shadow.");
        continue;
      }
      if (H.autoUpdate === !1 && H.needsUpdate === !1) continue;
      s.copy(H.mapSize);
      const fe = H.getFrameExtents();
      if (s.multiply(fe), r.copy(H.mapSize), (s.x > u || s.y > u) && (s.x > u && (r.x = Math.floor(u / fe.x), s.x = r.x * fe.x, H.mapSize.x = r.x), s.y > u && (r.y = Math.floor(u / fe.y), s.y = r.y * fe.y, H.mapSize.y = r.y)), H.map === null || V === !0 || Z === !0) {
        const me = this.type !== jn ? { minFilter: yn, magFilter: yn } : {};
        H.map !== null && H.map.dispose(), H.map = new qi(s.x, s.y, me), H.map.texture.name = ie.name + ".shadowMap", H.camera.updateProjectionMatrix();
      }
      n.setRenderTarget(H.map), n.clear();
      const xe = H.getViewportCount();
      for (let me = 0; me < xe; me++) {
        const de = H.getViewport(me);
        o.set(
          r.x * de.x,
          r.y * de.y,
          r.x * de.z,
          r.y * de.w
        ), L.viewport(o), H.updateMatrices(ie, me), i = H.getFrustum(), M(D, U, H.camera, ie, this.type);
      }
      H.isPointLightShadow !== !0 && this.type === jn && b(H, U), H.needsUpdate = !1;
    }
    d = this.type, m.needsUpdate = !1, n.setRenderTarget(y, S, P);
  };
  function b(w, D) {
    const U = e.update(x);
    f.defines.VSM_SAMPLES !== w.blurSamples && (f.defines.VSM_SAMPLES = w.blurSamples, p.defines.VSM_SAMPLES = w.blurSamples, f.needsUpdate = !0, p.needsUpdate = !0), w.mapPass === null && (w.mapPass = new qi(s.x, s.y)), f.uniforms.shadow_pass.value = w.map.texture, f.uniforms.resolution.value = w.mapSize, f.uniforms.radius.value = w.radius, n.setRenderTarget(w.mapPass), n.clear(), n.renderBufferDirect(D, null, U, f, x, null), p.uniforms.shadow_pass.value = w.mapPass.texture, p.uniforms.resolution.value = w.mapSize, p.uniforms.radius.value = w.radius, n.setRenderTarget(w.map), n.clear(), n.renderBufferDirect(D, null, U, p, x, null);
  }
  function A(w, D, U, y) {
    let S = null;
    const P = U.isPointLight === !0 ? w.customDistanceMaterial : w.customDepthMaterial;
    if (P !== void 0)
      S = P;
    else if (S = U.isPointLight === !0 ? l : a, n.localClippingEnabled && D.clipShadows === !0 && Array.isArray(D.clippingPlanes) && D.clippingPlanes.length !== 0 || D.displacementMap && D.displacementScale !== 0 || D.alphaMap && D.alphaTest > 0 || D.map && D.alphaTest > 0 || D.alphaToCoverage === !0) {
      const L = S.uuid, V = D.uuid;
      let Z = c[L];
      Z === void 0 && (Z = {}, c[L] = Z);
      let te = Z[V];
      te === void 0 && (te = S.clone(), Z[V] = te, D.addEventListener("dispose", R)), S = te;
    }
    if (S.visible = D.visible, S.wireframe = D.wireframe, y === jn ? S.side = D.shadowSide !== null ? D.shadowSide : D.side : S.side = D.shadowSide !== null ? D.shadowSide : h[D.side], S.alphaMap = D.alphaMap, S.alphaTest = D.alphaToCoverage === !0 ? 0.5 : D.alphaTest, S.map = D.map, S.clipShadows = D.clipShadows, S.clippingPlanes = D.clippingPlanes, S.clipIntersection = D.clipIntersection, S.displacementMap = D.displacementMap, S.displacementScale = D.displacementScale, S.displacementBias = D.displacementBias, S.wireframeLinewidth = D.wireframeLinewidth, S.linewidth = D.linewidth, U.isPointLight === !0 && S.isMeshDistanceMaterial === !0) {
      const L = n.properties.get(S);
      L.light = U;
    }
    return S;
  }
  function M(w, D, U, y, S) {
    if (w.visible === !1) return;
    if (w.layers.test(D.layers) && (w.isMesh || w.isLine || w.isPoints) && (w.castShadow || w.receiveShadow && S === jn) && (!w.frustumCulled || i.intersectsObject(w))) {
      w.modelViewMatrix.multiplyMatrices(U.matrixWorldInverse, w.matrixWorld);
      const V = e.update(w), Z = w.material;
      if (Array.isArray(Z)) {
        const te = V.groups;
        for (let $ = 0, ie = te.length; $ < ie; $++) {
          const H = te[$], fe = Z[H.materialIndex];
          if (fe && fe.visible) {
            const xe = A(w, fe, y, S);
            w.onBeforeShadow(n, w, D, U, V, xe, H), n.renderBufferDirect(U, null, V, xe, w, H), w.onAfterShadow(n, w, D, U, V, xe, H);
          }
        }
      } else if (Z.visible) {
        const te = A(w, Z, y, S);
        w.onBeforeShadow(n, w, D, U, V, te, null), n.renderBufferDirect(U, null, V, te, w, null), w.onAfterShadow(n, w, D, U, V, te, null);
      }
    }
    const L = w.children;
    for (let V = 0, Z = L.length; V < Z; V++)
      M(L[V], D, U, y, S);
  }
  function R(w) {
    w.target.removeEventListener("dispose", R);
    for (const U in c) {
      const y = c[U], S = w.target.uuid;
      S in y && (y[S].dispose(), delete y[S]);
    }
  }
}
const GS = {
  [pl]: ml,
  [_l]: xl,
  [gl]: Ml,
  [Us]: vl,
  [ml]: pl,
  [xl]: _l,
  [Ml]: gl,
  [vl]: Us
};
function WS(n, e) {
  function t() {
    let F = !1;
    const he = new lt();
    let pe = null;
    const Re = new lt(0, 0, 0, 0);
    return {
      setMask: function(ce) {
        pe !== ce && !F && (n.colorMask(ce, ce, ce, ce), pe = ce);
      },
      setLocked: function(ce) {
        F = ce;
      },
      setClear: function(ce, se, Ie, Ge, ht) {
        ht === !0 && (ce *= Ge, se *= Ge, Ie *= Ge), he.set(ce, se, Ie, Ge), Re.equals(he) === !1 && (n.clearColor(ce, se, Ie, Ge), Re.copy(he));
      },
      reset: function() {
        F = !1, pe = null, Re.set(-1, 0, 0, 0);
      }
    };
  }
  function i() {
    let F = !1, he = !1, pe = null, Re = null, ce = null;
    return {
      setReversed: function(se) {
        if (he !== se) {
          const Ie = e.get("EXT_clip_control");
          se ? Ie.clipControlEXT(Ie.LOWER_LEFT_EXT, Ie.ZERO_TO_ONE_EXT) : Ie.clipControlEXT(Ie.LOWER_LEFT_EXT, Ie.NEGATIVE_ONE_TO_ONE_EXT), he = se;
          const Ge = ce;
          ce = null, this.setClear(Ge);
        }
      },
      getReversed: function() {
        return he;
      },
      setTest: function(se) {
        se ? re(n.DEPTH_TEST) : Ae(n.DEPTH_TEST);
      },
      setMask: function(se) {
        pe !== se && !F && (n.depthMask(se), pe = se);
      },
      setFunc: function(se) {
        if (he && (se = GS[se]), Re !== se) {
          switch (se) {
            case pl:
              n.depthFunc(n.NEVER);
              break;
            case ml:
              n.depthFunc(n.ALWAYS);
              break;
            case _l:
              n.depthFunc(n.LESS);
              break;
            case Us:
              n.depthFunc(n.LEQUAL);
              break;
            case gl:
              n.depthFunc(n.EQUAL);
              break;
            case vl:
              n.depthFunc(n.GEQUAL);
              break;
            case xl:
              n.depthFunc(n.GREATER);
              break;
            case Ml:
              n.depthFunc(n.NOTEQUAL);
              break;
            default:
              n.depthFunc(n.LEQUAL);
          }
          Re = se;
        }
      },
      setLocked: function(se) {
        F = se;
      },
      setClear: function(se) {
        ce !== se && (he && (se = 1 - se), n.clearDepth(se), ce = se);
      },
      reset: function() {
        F = !1, pe = null, Re = null, ce = null, he = !1;
      }
    };
  }
  function s() {
    let F = !1, he = null, pe = null, Re = null, ce = null, se = null, Ie = null, Ge = null, ht = null;
    return {
      setTest: function(nt) {
        F || (nt ? re(n.STENCIL_TEST) : Ae(n.STENCIL_TEST));
      },
      setMask: function(nt) {
        he !== nt && !F && (n.stencilMask(nt), he = nt);
      },
      setFunc: function(nt, Hn, bn) {
        (pe !== nt || Re !== Hn || ce !== bn) && (n.stencilFunc(nt, Hn, bn), pe = nt, Re = Hn, ce = bn);
      },
      setOp: function(nt, Hn, bn) {
        (se !== nt || Ie !== Hn || Ge !== bn) && (n.stencilOp(nt, Hn, bn), se = nt, Ie = Hn, Ge = bn);
      },
      setLocked: function(nt) {
        F = nt;
      },
      setClear: function(nt) {
        ht !== nt && (n.clearStencil(nt), ht = nt);
      },
      reset: function() {
        F = !1, he = null, pe = null, Re = null, ce = null, se = null, Ie = null, Ge = null, ht = null;
      }
    };
  }
  const r = new t(), o = new i(), a = new s(), l = /* @__PURE__ */ new WeakMap(), c = /* @__PURE__ */ new WeakMap();
  let u = {}, h = {}, f = /* @__PURE__ */ new WeakMap(), p = [], v = null, x = !1, m = null, d = null, b = null, A = null, M = null, R = null, w = null, D = new We(0, 0, 0), U = 0, y = !1, S = null, P = null, L = null, V = null, Z = null;
  const te = n.getParameter(n.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
  let $ = !1, ie = 0;
  const H = n.getParameter(n.VERSION);
  H.indexOf("WebGL") !== -1 ? (ie = parseFloat(/^WebGL (\d)/.exec(H)[1]), $ = ie >= 1) : H.indexOf("OpenGL ES") !== -1 && (ie = parseFloat(/^OpenGL ES (\d)/.exec(H)[1]), $ = ie >= 2);
  let fe = null, xe = {};
  const me = n.getParameter(n.SCISSOR_BOX), de = n.getParameter(n.VIEWPORT), Le = new lt().fromArray(me), tt = new lt().fromArray(de);
  function Ze(F, he, pe, Re) {
    const ce = new Uint8Array(4), se = n.createTexture();
    n.bindTexture(F, se), n.texParameteri(F, n.TEXTURE_MIN_FILTER, n.NEAREST), n.texParameteri(F, n.TEXTURE_MAG_FILTER, n.NEAREST);
    for (let Ie = 0; Ie < pe; Ie++)
      F === n.TEXTURE_3D || F === n.TEXTURE_2D_ARRAY ? n.texImage3D(he, 0, n.RGBA, 1, 1, Re, 0, n.RGBA, n.UNSIGNED_BYTE, ce) : n.texImage2D(he + Ie, 0, n.RGBA, 1, 1, 0, n.RGBA, n.UNSIGNED_BYTE, ce);
    return se;
  }
  const ne = {};
  ne[n.TEXTURE_2D] = Ze(n.TEXTURE_2D, n.TEXTURE_2D, 1), ne[n.TEXTURE_CUBE_MAP] = Ze(n.TEXTURE_CUBE_MAP, n.TEXTURE_CUBE_MAP_POSITIVE_X, 6), ne[n.TEXTURE_2D_ARRAY] = Ze(n.TEXTURE_2D_ARRAY, n.TEXTURE_2D_ARRAY, 1, 1), ne[n.TEXTURE_3D] = Ze(n.TEXTURE_3D, n.TEXTURE_3D, 1, 1), r.setClear(0, 0, 0, 1), o.setClear(1), a.setClear(0), re(n.DEPTH_TEST), o.setFunc(Us), X(!1), z(Tu), re(n.CULL_FACE), W(vi);
  function re(F) {
    u[F] !== !0 && (n.enable(F), u[F] = !0);
  }
  function Ae(F) {
    u[F] !== !1 && (n.disable(F), u[F] = !1);
  }
  function Oe(F, he) {
    return h[F] !== he ? (n.bindFramebuffer(F, he), h[F] = he, F === n.DRAW_FRAMEBUFFER && (h[n.FRAMEBUFFER] = he), F === n.FRAMEBUFFER && (h[n.DRAW_FRAMEBUFFER] = he), !0) : !1;
  }
  function Pe(F, he) {
    let pe = p, Re = !1;
    if (F) {
      pe = f.get(he), pe === void 0 && (pe = [], f.set(he, pe));
      const ce = F.textures;
      if (pe.length !== ce.length || pe[0] !== n.COLOR_ATTACHMENT0) {
        for (let se = 0, Ie = ce.length; se < Ie; se++)
          pe[se] = n.COLOR_ATTACHMENT0 + se;
        pe.length = ce.length, Re = !0;
      }
    } else
      pe[0] !== n.BACK && (pe[0] = n.BACK, Re = !0);
    Re && n.drawBuffers(pe);
  }
  function $e(F) {
    return v !== F ? (n.useProgram(F), v = F, !0) : !1;
  }
  const C = {
    [Bi]: n.FUNC_ADD,
    [X_]: n.FUNC_SUBTRACT,
    [Y_]: n.FUNC_REVERSE_SUBTRACT
  };
  C[q_] = n.MIN, C[j_] = n.MAX;
  const g = {
    [K_]: n.ZERO,
    [$_]: n.ONE,
    [Z_]: n.SRC_COLOR,
    [fl]: n.SRC_ALPHA,
    [ig]: n.SRC_ALPHA_SATURATE,
    [tg]: n.DST_COLOR,
    [Q_]: n.DST_ALPHA,
    [J_]: n.ONE_MINUS_SRC_COLOR,
    [dl]: n.ONE_MINUS_SRC_ALPHA,
    [ng]: n.ONE_MINUS_DST_COLOR,
    [eg]: n.ONE_MINUS_DST_ALPHA,
    [sg]: n.CONSTANT_COLOR,
    [rg]: n.ONE_MINUS_CONSTANT_COLOR,
    [og]: n.CONSTANT_ALPHA,
    [ag]: n.ONE_MINUS_CONSTANT_ALPHA
  };
  function W(F, he, pe, Re, ce, se, Ie, Ge, ht, nt) {
    if (F === vi) {
      x === !0 && (Ae(n.BLEND), x = !1);
      return;
    }
    if (x === !1 && (re(n.BLEND), x = !0), F !== W_) {
      if (F !== m || nt !== y) {
        if ((d !== Bi || M !== Bi) && (n.blendEquation(n.FUNC_ADD), d = Bi, M = Bi), nt)
          switch (F) {
            case Ds:
              n.blendFuncSeparate(n.ONE, n.ONE_MINUS_SRC_ALPHA, n.ONE, n.ONE_MINUS_SRC_ALPHA);
              break;
            case bu:
              n.blendFunc(n.ONE, n.ONE);
              break;
            case Au:
              n.blendFuncSeparate(n.ZERO, n.ONE_MINUS_SRC_COLOR, n.ZERO, n.ONE);
              break;
            case wu:
              n.blendFuncSeparate(n.DST_COLOR, n.ONE_MINUS_SRC_ALPHA, n.ZERO, n.ONE);
              break;
            default:
              console.error("THREE.WebGLState: Invalid blending: ", F);
              break;
          }
        else
          switch (F) {
            case Ds:
              n.blendFuncSeparate(n.SRC_ALPHA, n.ONE_MINUS_SRC_ALPHA, n.ONE, n.ONE_MINUS_SRC_ALPHA);
              break;
            case bu:
              n.blendFuncSeparate(n.SRC_ALPHA, n.ONE, n.ONE, n.ONE);
              break;
            case Au:
              console.error("THREE.WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");
              break;
            case wu:
              console.error("THREE.WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");
              break;
            default:
              console.error("THREE.WebGLState: Invalid blending: ", F);
              break;
          }
        b = null, A = null, R = null, w = null, D.set(0, 0, 0), U = 0, m = F, y = nt;
      }
      return;
    }
    ce = ce || he, se = se || pe, Ie = Ie || Re, (he !== d || ce !== M) && (n.blendEquationSeparate(C[he], C[ce]), d = he, M = ce), (pe !== b || Re !== A || se !== R || Ie !== w) && (n.blendFuncSeparate(g[pe], g[Re], g[se], g[Ie]), b = pe, A = Re, R = se, w = Ie), (Ge.equals(D) === !1 || ht !== U) && (n.blendColor(Ge.r, Ge.g, Ge.b, ht), D.copy(Ge), U = ht), m = F, y = !1;
  }
  function j(F, he) {
    F.side === Qn ? Ae(n.CULL_FACE) : re(n.CULL_FACE);
    let pe = F.side === Wt;
    he && (pe = !pe), X(pe), F.blending === Ds && F.transparent === !1 ? W(vi) : W(F.blending, F.blendEquation, F.blendSrc, F.blendDst, F.blendEquationAlpha, F.blendSrcAlpha, F.blendDstAlpha, F.blendColor, F.blendAlpha, F.premultipliedAlpha), o.setFunc(F.depthFunc), o.setTest(F.depthTest), o.setMask(F.depthWrite), r.setMask(F.colorWrite);
    const Re = F.stencilWrite;
    a.setTest(Re), Re && (a.setMask(F.stencilWriteMask), a.setFunc(F.stencilFunc, F.stencilRef, F.stencilFuncMask), a.setOp(F.stencilFail, F.stencilZFail, F.stencilZPass)), q(F.polygonOffset, F.polygonOffsetFactor, F.polygonOffsetUnits), F.alphaToCoverage === !0 ? re(n.SAMPLE_ALPHA_TO_COVERAGE) : Ae(n.SAMPLE_ALPHA_TO_COVERAGE);
  }
  function X(F) {
    S !== F && (F ? n.frontFace(n.CW) : n.frontFace(n.CCW), S = F);
  }
  function z(F) {
    F !== V_ ? (re(n.CULL_FACE), F !== P && (F === Tu ? n.cullFace(n.BACK) : F === k_ ? n.cullFace(n.FRONT) : n.cullFace(n.FRONT_AND_BACK))) : Ae(n.CULL_FACE), P = F;
  }
  function ae(F) {
    F !== L && ($ && n.lineWidth(F), L = F);
  }
  function q(F, he, pe) {
    F ? (re(n.POLYGON_OFFSET_FILL), (V !== he || Z !== pe) && (n.polygonOffset(he, pe), V = he, Z = pe)) : Ae(n.POLYGON_OFFSET_FILL);
  }
  function Q(F) {
    F ? re(n.SCISSOR_TEST) : Ae(n.SCISSOR_TEST);
  }
  function ee(F) {
    F === void 0 && (F = n.TEXTURE0 + te - 1), fe !== F && (n.activeTexture(F), fe = F);
  }
  function Se(F, he, pe) {
    pe === void 0 && (fe === null ? pe = n.TEXTURE0 + te - 1 : pe = fe);
    let Re = xe[pe];
    Re === void 0 && (Re = { type: void 0, texture: void 0 }, xe[pe] = Re), (Re.type !== F || Re.texture !== he) && (fe !== pe && (n.activeTexture(pe), fe = pe), n.bindTexture(F, he || ne[F]), Re.type = F, Re.texture = he);
  }
  function E() {
    const F = xe[fe];
    F !== void 0 && F.type !== void 0 && (n.bindTexture(F.type, null), F.type = void 0, F.texture = void 0);
  }
  function _() {
    try {
      n.compressedTexImage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function I() {
    try {
      n.compressedTexImage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function k() {
    try {
      n.texSubImage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function J() {
    try {
      n.texSubImage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function G() {
    try {
      n.compressedTexSubImage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function _e() {
    try {
      n.compressedTexSubImage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function oe() {
    try {
      n.texStorage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function Ee() {
    try {
      n.texStorage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function Te() {
    try {
      n.texImage2D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function le() {
    try {
      n.texImage3D(...arguments);
    } catch (F) {
      console.error("THREE.WebGLState:", F);
    }
  }
  function Me(F) {
    Le.equals(F) === !1 && (n.scissor(F.x, F.y, F.z, F.w), Le.copy(F));
  }
  function Ce(F) {
    tt.equals(F) === !1 && (n.viewport(F.x, F.y, F.z, F.w), tt.copy(F));
  }
  function be(F, he) {
    let pe = c.get(he);
    pe === void 0 && (pe = /* @__PURE__ */ new WeakMap(), c.set(he, pe));
    let Re = pe.get(F);
    Re === void 0 && (Re = n.getUniformBlockIndex(he, F.name), pe.set(F, Re));
  }
  function ge(F, he) {
    const Re = c.get(he).get(F);
    l.get(he) !== Re && (n.uniformBlockBinding(he, Re, F.__bindingPointIndex), l.set(he, Re));
  }
  function ke() {
    n.disable(n.BLEND), n.disable(n.CULL_FACE), n.disable(n.DEPTH_TEST), n.disable(n.POLYGON_OFFSET_FILL), n.disable(n.SCISSOR_TEST), n.disable(n.STENCIL_TEST), n.disable(n.SAMPLE_ALPHA_TO_COVERAGE), n.blendEquation(n.FUNC_ADD), n.blendFunc(n.ONE, n.ZERO), n.blendFuncSeparate(n.ONE, n.ZERO, n.ONE, n.ZERO), n.blendColor(0, 0, 0, 0), n.colorMask(!0, !0, !0, !0), n.clearColor(0, 0, 0, 0), n.depthMask(!0), n.depthFunc(n.LESS), o.setReversed(!1), n.clearDepth(1), n.stencilMask(4294967295), n.stencilFunc(n.ALWAYS, 0, 4294967295), n.stencilOp(n.KEEP, n.KEEP, n.KEEP), n.clearStencil(0), n.cullFace(n.BACK), n.frontFace(n.CCW), n.polygonOffset(0, 0), n.activeTexture(n.TEXTURE0), n.bindFramebuffer(n.FRAMEBUFFER, null), n.bindFramebuffer(n.DRAW_FRAMEBUFFER, null), n.bindFramebuffer(n.READ_FRAMEBUFFER, null), n.useProgram(null), n.lineWidth(1), n.scissor(0, 0, n.canvas.width, n.canvas.height), n.viewport(0, 0, n.canvas.width, n.canvas.height), u = {}, fe = null, xe = {}, h = {}, f = /* @__PURE__ */ new WeakMap(), p = [], v = null, x = !1, m = null, d = null, b = null, A = null, M = null, R = null, w = null, D = new We(0, 0, 0), U = 0, y = !1, S = null, P = null, L = null, V = null, Z = null, Le.set(0, 0, n.canvas.width, n.canvas.height), tt.set(0, 0, n.canvas.width, n.canvas.height), r.reset(), o.reset(), a.reset();
  }
  return {
    buffers: {
      color: r,
      depth: o,
      stencil: a
    },
    enable: re,
    disable: Ae,
    bindFramebuffer: Oe,
    drawBuffers: Pe,
    useProgram: $e,
    setBlending: W,
    setMaterial: j,
    setFlipSided: X,
    setCullFace: z,
    setLineWidth: ae,
    setPolygonOffset: q,
    setScissorTest: Q,
    activeTexture: ee,
    bindTexture: Se,
    unbindTexture: E,
    compressedTexImage2D: _,
    compressedTexImage3D: I,
    texImage2D: Te,
    texImage3D: le,
    updateUBOMapping: be,
    uniformBlockBinding: ge,
    texStorage2D: oe,
    texStorage3D: Ee,
    texSubImage2D: k,
    texSubImage3D: J,
    compressedTexSubImage2D: G,
    compressedTexSubImage3D: _e,
    scissor: Me,
    viewport: Ce,
    reset: ke
  };
}
function XS(n, e, t, i, s, r, o) {
  const a = e.has("WEBGL_multisampled_render_to_texture") ? e.get("WEBGL_multisampled_render_to_texture") : null, l = typeof navigator > "u" ? !1 : /OculusBrowser/g.test(navigator.userAgent), c = new Ve(), u = /* @__PURE__ */ new WeakMap();
  let h;
  const f = /* @__PURE__ */ new WeakMap();
  let p = !1;
  try {
    p = typeof OffscreenCanvas < "u" && new OffscreenCanvas(1, 1).getContext("2d") !== null;
  } catch {
  }
  function v(E, _) {
    return p ? (
      // eslint-disable-next-line compat/compat
      new OffscreenCanvas(E, _)
    ) : Ho("canvas");
  }
  function x(E, _, I) {
    let k = 1;
    const J = Se(E);
    if ((J.width > I || J.height > I) && (k = I / Math.max(J.width, J.height)), k < 1)
      if (typeof HTMLImageElement < "u" && E instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && E instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && E instanceof ImageBitmap || typeof VideoFrame < "u" && E instanceof VideoFrame) {
        const G = Math.floor(k * J.width), _e = Math.floor(k * J.height);
        h === void 0 && (h = v(G, _e));
        const oe = _ ? v(G, _e) : h;
        return oe.width = G, oe.height = _e, oe.getContext("2d").drawImage(E, 0, 0, G, _e), console.warn("THREE.WebGLRenderer: Texture has been resized from (" + J.width + "x" + J.height + ") to (" + G + "x" + _e + ")."), oe;
      } else
        return "data" in E && console.warn("THREE.WebGLRenderer: Image in DataTexture is too big (" + J.width + "x" + J.height + ")."), E;
    return E;
  }
  function m(E) {
    return E.generateMipmaps;
  }
  function d(E) {
    n.generateMipmap(E);
  }
  function b(E) {
    return E.isWebGLCubeRenderTarget ? n.TEXTURE_CUBE_MAP : E.isWebGL3DRenderTarget ? n.TEXTURE_3D : E.isWebGLArrayRenderTarget || E.isCompressedArrayTexture ? n.TEXTURE_2D_ARRAY : n.TEXTURE_2D;
  }
  function A(E, _, I, k, J = !1) {
    if (E !== null) {
      if (n[E] !== void 0) return n[E];
      console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '" + E + "'");
    }
    let G = _;
    if (_ === n.RED && (I === n.FLOAT && (G = n.R32F), I === n.HALF_FLOAT && (G = n.R16F), I === n.UNSIGNED_BYTE && (G = n.R8)), _ === n.RED_INTEGER && (I === n.UNSIGNED_BYTE && (G = n.R8UI), I === n.UNSIGNED_SHORT && (G = n.R16UI), I === n.UNSIGNED_INT && (G = n.R32UI), I === n.BYTE && (G = n.R8I), I === n.SHORT && (G = n.R16I), I === n.INT && (G = n.R32I)), _ === n.RG && (I === n.FLOAT && (G = n.RG32F), I === n.HALF_FLOAT && (G = n.RG16F), I === n.UNSIGNED_BYTE && (G = n.RG8)), _ === n.RG_INTEGER && (I === n.UNSIGNED_BYTE && (G = n.RG8UI), I === n.UNSIGNED_SHORT && (G = n.RG16UI), I === n.UNSIGNED_INT && (G = n.RG32UI), I === n.BYTE && (G = n.RG8I), I === n.SHORT && (G = n.RG16I), I === n.INT && (G = n.RG32I)), _ === n.RGB_INTEGER && (I === n.UNSIGNED_BYTE && (G = n.RGB8UI), I === n.UNSIGNED_SHORT && (G = n.RGB16UI), I === n.UNSIGNED_INT && (G = n.RGB32UI), I === n.BYTE && (G = n.RGB8I), I === n.SHORT && (G = n.RGB16I), I === n.INT && (G = n.RGB32I)), _ === n.RGBA_INTEGER && (I === n.UNSIGNED_BYTE && (G = n.RGBA8UI), I === n.UNSIGNED_SHORT && (G = n.RGBA16UI), I === n.UNSIGNED_INT && (G = n.RGBA32UI), I === n.BYTE && (G = n.RGBA8I), I === n.SHORT && (G = n.RGBA16I), I === n.INT && (G = n.RGBA32I)), _ === n.RGB && (I === n.UNSIGNED_INT_5_9_9_9_REV && (G = n.RGB9_E5), I === n.UNSIGNED_INT_10F_11F_11F_REV && (G = n.R11F_G11F_B10F)), _ === n.RGBA) {
      const _e = J ? Bo : Qe.getTransfer(k);
      I === n.FLOAT && (G = n.RGBA32F), I === n.HALF_FLOAT && (G = n.RGBA16F), I === n.UNSIGNED_BYTE && (G = _e === ot ? n.SRGB8_ALPHA8 : n.RGBA8), I === n.UNSIGNED_SHORT_4_4_4_4 && (G = n.RGBA4), I === n.UNSIGNED_SHORT_5_5_5_1 && (G = n.RGB5_A1);
    }
    return (G === n.R16F || G === n.R32F || G === n.RG16F || G === n.RG32F || G === n.RGBA16F || G === n.RGBA32F) && e.get("EXT_color_buffer_float"), G;
  }
  function M(E, _) {
    let I;
    return E ? _ === null || _ === Xi || _ === br ? I = n.DEPTH24_STENCIL8 : _ === ei ? I = n.DEPTH32F_STENCIL8 : _ === Tr && (I = n.DEPTH24_STENCIL8, console.warn("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")) : _ === null || _ === Xi || _ === br ? I = n.DEPTH_COMPONENT24 : _ === ei ? I = n.DEPTH_COMPONENT32F : _ === Tr && (I = n.DEPTH_COMPONENT16), I;
  }
  function R(E, _) {
    return m(E) === !0 || E.isFramebufferTexture && E.minFilter !== yn && E.minFilter !== Un ? Math.log2(Math.max(_.width, _.height)) + 1 : E.mipmaps !== void 0 && E.mipmaps.length > 0 ? E.mipmaps.length : E.isCompressedTexture && Array.isArray(E.image) ? _.mipmaps.length : 1;
  }
  function w(E) {
    const _ = E.target;
    _.removeEventListener("dispose", w), U(_), _.isVideoTexture && u.delete(_);
  }
  function D(E) {
    const _ = E.target;
    _.removeEventListener("dispose", D), S(_);
  }
  function U(E) {
    const _ = i.get(E);
    if (_.__webglInit === void 0) return;
    const I = E.source, k = f.get(I);
    if (k) {
      const J = k[_.__cacheKey];
      J.usedTimes--, J.usedTimes === 0 && y(E), Object.keys(k).length === 0 && f.delete(I);
    }
    i.remove(E);
  }
  function y(E) {
    const _ = i.get(E);
    n.deleteTexture(_.__webglTexture);
    const I = E.source, k = f.get(I);
    delete k[_.__cacheKey], o.memory.textures--;
  }
  function S(E) {
    const _ = i.get(E);
    if (E.depthTexture && (E.depthTexture.dispose(), i.remove(E.depthTexture)), E.isWebGLCubeRenderTarget)
      for (let k = 0; k < 6; k++) {
        if (Array.isArray(_.__webglFramebuffer[k]))
          for (let J = 0; J < _.__webglFramebuffer[k].length; J++) n.deleteFramebuffer(_.__webglFramebuffer[k][J]);
        else
          n.deleteFramebuffer(_.__webglFramebuffer[k]);
        _.__webglDepthbuffer && n.deleteRenderbuffer(_.__webglDepthbuffer[k]);
      }
    else {
      if (Array.isArray(_.__webglFramebuffer))
        for (let k = 0; k < _.__webglFramebuffer.length; k++) n.deleteFramebuffer(_.__webglFramebuffer[k]);
      else
        n.deleteFramebuffer(_.__webglFramebuffer);
      if (_.__webglDepthbuffer && n.deleteRenderbuffer(_.__webglDepthbuffer), _.__webglMultisampledFramebuffer && n.deleteFramebuffer(_.__webglMultisampledFramebuffer), _.__webglColorRenderbuffer)
        for (let k = 0; k < _.__webglColorRenderbuffer.length; k++)
          _.__webglColorRenderbuffer[k] && n.deleteRenderbuffer(_.__webglColorRenderbuffer[k]);
      _.__webglDepthRenderbuffer && n.deleteRenderbuffer(_.__webglDepthRenderbuffer);
    }
    const I = E.textures;
    for (let k = 0, J = I.length; k < J; k++) {
      const G = i.get(I[k]);
      G.__webglTexture && (n.deleteTexture(G.__webglTexture), o.memory.textures--), i.remove(I[k]);
    }
    i.remove(E);
  }
  let P = 0;
  function L() {
    P = 0;
  }
  function V() {
    const E = P;
    return E >= s.maxTextures && console.warn("THREE.WebGLTextures: Trying to use " + E + " texture units while this GPU supports only " + s.maxTextures), P += 1, E;
  }
  function Z(E) {
    const _ = [];
    return _.push(E.wrapS), _.push(E.wrapT), _.push(E.wrapR || 0), _.push(E.magFilter), _.push(E.minFilter), _.push(E.anisotropy), _.push(E.internalFormat), _.push(E.format), _.push(E.type), _.push(E.generateMipmaps), _.push(E.premultiplyAlpha), _.push(E.flipY), _.push(E.unpackAlignment), _.push(E.colorSpace), _.join();
  }
  function te(E, _) {
    const I = i.get(E);
    if (E.isVideoTexture && Q(E), E.isRenderTargetTexture === !1 && E.isExternalTexture !== !0 && E.version > 0 && I.__version !== E.version) {
      const k = E.image;
      if (k === null)
        console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");
      else if (k.complete === !1)
        console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");
      else {
        ne(I, E, _);
        return;
      }
    } else E.isExternalTexture && (I.__webglTexture = E.sourceTexture ? E.sourceTexture : null);
    t.bindTexture(n.TEXTURE_2D, I.__webglTexture, n.TEXTURE0 + _);
  }
  function $(E, _) {
    const I = i.get(E);
    if (E.isRenderTargetTexture === !1 && E.version > 0 && I.__version !== E.version) {
      ne(I, E, _);
      return;
    }
    t.bindTexture(n.TEXTURE_2D_ARRAY, I.__webglTexture, n.TEXTURE0 + _);
  }
  function ie(E, _) {
    const I = i.get(E);
    if (E.isRenderTargetTexture === !1 && E.version > 0 && I.__version !== E.version) {
      ne(I, E, _);
      return;
    }
    t.bindTexture(n.TEXTURE_3D, I.__webglTexture, n.TEXTURE0 + _);
  }
  function H(E, _) {
    const I = i.get(E);
    if (E.version > 0 && I.__version !== E.version) {
      re(I, E, _);
      return;
    }
    t.bindTexture(n.TEXTURE_CUBE_MAP, I.__webglTexture, n.TEXTURE0 + _);
  }
  const fe = {
    [El]: n.REPEAT,
    [Hi]: n.CLAMP_TO_EDGE,
    [Tl]: n.MIRRORED_REPEAT
  }, xe = {
    [yn]: n.NEAREST,
    [_g]: n.NEAREST_MIPMAP_NEAREST,
    [Vr]: n.NEAREST_MIPMAP_LINEAR,
    [Un]: n.LINEAR,
    [xa]: n.LINEAR_MIPMAP_NEAREST,
    [Vi]: n.LINEAR_MIPMAP_LINEAR
  }, me = {
    [Mg]: n.NEVER,
    [Ag]: n.ALWAYS,
    [Sg]: n.LESS,
    [hd]: n.LEQUAL,
    [yg]: n.EQUAL,
    [bg]: n.GEQUAL,
    [Eg]: n.GREATER,
    [Tg]: n.NOTEQUAL
  };
  function de(E, _) {
    if (_.type === ei && e.has("OES_texture_float_linear") === !1 && (_.magFilter === Un || _.magFilter === xa || _.magFilter === Vr || _.magFilter === Vi || _.minFilter === Un || _.minFilter === xa || _.minFilter === Vr || _.minFilter === Vi) && console.warn("THREE.WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."), n.texParameteri(E, n.TEXTURE_WRAP_S, fe[_.wrapS]), n.texParameteri(E, n.TEXTURE_WRAP_T, fe[_.wrapT]), (E === n.TEXTURE_3D || E === n.TEXTURE_2D_ARRAY) && n.texParameteri(E, n.TEXTURE_WRAP_R, fe[_.wrapR]), n.texParameteri(E, n.TEXTURE_MAG_FILTER, xe[_.magFilter]), n.texParameteri(E, n.TEXTURE_MIN_FILTER, xe[_.minFilter]), _.compareFunction && (n.texParameteri(E, n.TEXTURE_COMPARE_MODE, n.COMPARE_REF_TO_TEXTURE), n.texParameteri(E, n.TEXTURE_COMPARE_FUNC, me[_.compareFunction])), e.has("EXT_texture_filter_anisotropic") === !0) {
      if (_.magFilter === yn || _.minFilter !== Vr && _.minFilter !== Vi || _.type === ei && e.has("OES_texture_float_linear") === !1) return;
      if (_.anisotropy > 1 || i.get(_).__currentAnisotropy) {
        const I = e.get("EXT_texture_filter_anisotropic");
        n.texParameterf(E, I.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(_.anisotropy, s.getMaxAnisotropy())), i.get(_).__currentAnisotropy = _.anisotropy;
      }
    }
  }
  function Le(E, _) {
    let I = !1;
    E.__webglInit === void 0 && (E.__webglInit = !0, _.addEventListener("dispose", w));
    const k = _.source;
    let J = f.get(k);
    J === void 0 && (J = {}, f.set(k, J));
    const G = Z(_);
    if (G !== E.__cacheKey) {
      J[G] === void 0 && (J[G] = {
        texture: n.createTexture(),
        usedTimes: 0
      }, o.memory.textures++, I = !0), J[G].usedTimes++;
      const _e = J[E.__cacheKey];
      _e !== void 0 && (J[E.__cacheKey].usedTimes--, _e.usedTimes === 0 && y(_)), E.__cacheKey = G, E.__webglTexture = J[G].texture;
    }
    return I;
  }
  function tt(E, _, I) {
    return Math.floor(Math.floor(E / I) / _);
  }
  function Ze(E, _, I, k) {
    const G = E.updateRanges;
    if (G.length === 0)
      t.texSubImage2D(n.TEXTURE_2D, 0, 0, 0, _.width, _.height, I, k, _.data);
    else {
      G.sort((le, Me) => le.start - Me.start);
      let _e = 0;
      for (let le = 1; le < G.length; le++) {
        const Me = G[_e], Ce = G[le], be = Me.start + Me.count, ge = tt(Ce.start, _.width, 4), ke = tt(Me.start, _.width, 4);
        Ce.start <= be + 1 && ge === ke && tt(Ce.start + Ce.count - 1, _.width, 4) === ge ? Me.count = Math.max(
          Me.count,
          Ce.start + Ce.count - Me.start
        ) : (++_e, G[_e] = Ce);
      }
      G.length = _e + 1;
      const oe = n.getParameter(n.UNPACK_ROW_LENGTH), Ee = n.getParameter(n.UNPACK_SKIP_PIXELS), Te = n.getParameter(n.UNPACK_SKIP_ROWS);
      n.pixelStorei(n.UNPACK_ROW_LENGTH, _.width);
      for (let le = 0, Me = G.length; le < Me; le++) {
        const Ce = G[le], be = Math.floor(Ce.start / 4), ge = Math.ceil(Ce.count / 4), ke = be % _.width, F = Math.floor(be / _.width), he = ge, pe = 1;
        n.pixelStorei(n.UNPACK_SKIP_PIXELS, ke), n.pixelStorei(n.UNPACK_SKIP_ROWS, F), t.texSubImage2D(n.TEXTURE_2D, 0, ke, F, he, pe, I, k, _.data);
      }
      E.clearUpdateRanges(), n.pixelStorei(n.UNPACK_ROW_LENGTH, oe), n.pixelStorei(n.UNPACK_SKIP_PIXELS, Ee), n.pixelStorei(n.UNPACK_SKIP_ROWS, Te);
    }
  }
  function ne(E, _, I) {
    let k = n.TEXTURE_2D;
    (_.isDataArrayTexture || _.isCompressedArrayTexture) && (k = n.TEXTURE_2D_ARRAY), _.isData3DTexture && (k = n.TEXTURE_3D);
    const J = Le(E, _), G = _.source;
    t.bindTexture(k, E.__webglTexture, n.TEXTURE0 + I);
    const _e = i.get(G);
    if (G.version !== _e.__version || J === !0) {
      t.activeTexture(n.TEXTURE0 + I);
      const oe = Qe.getPrimaries(Qe.workingColorSpace), Ee = _.colorSpace === gi ? null : Qe.getPrimaries(_.colorSpace), Te = _.colorSpace === gi || oe === Ee ? n.NONE : n.BROWSER_DEFAULT_WEBGL;
      n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL, _.flipY), n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL, _.premultiplyAlpha), n.pixelStorei(n.UNPACK_ALIGNMENT, _.unpackAlignment), n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL, Te);
      let le = x(_.image, !1, s.maxTextureSize);
      le = ee(_, le);
      const Me = r.convert(_.format, _.colorSpace), Ce = r.convert(_.type);
      let be = A(_.internalFormat, Me, Ce, _.colorSpace, _.isVideoTexture);
      de(k, _);
      let ge;
      const ke = _.mipmaps, F = _.isVideoTexture !== !0, he = _e.__version === void 0 || J === !0, pe = G.dataReady, Re = R(_, le);
      if (_.isDepthTexture)
        be = M(_.format === wr, _.type), he && (F ? t.texStorage2D(n.TEXTURE_2D, 1, be, le.width, le.height) : t.texImage2D(n.TEXTURE_2D, 0, be, le.width, le.height, 0, Me, Ce, null));
      else if (_.isDataTexture)
        if (ke.length > 0) {
          F && he && t.texStorage2D(n.TEXTURE_2D, Re, be, ke[0].width, ke[0].height);
          for (let ce = 0, se = ke.length; ce < se; ce++)
            ge = ke[ce], F ? pe && t.texSubImage2D(n.TEXTURE_2D, ce, 0, 0, ge.width, ge.height, Me, Ce, ge.data) : t.texImage2D(n.TEXTURE_2D, ce, be, ge.width, ge.height, 0, Me, Ce, ge.data);
          _.generateMipmaps = !1;
        } else
          F ? (he && t.texStorage2D(n.TEXTURE_2D, Re, be, le.width, le.height), pe && Ze(_, le, Me, Ce)) : t.texImage2D(n.TEXTURE_2D, 0, be, le.width, le.height, 0, Me, Ce, le.data);
      else if (_.isCompressedTexture)
        if (_.isCompressedArrayTexture) {
          F && he && t.texStorage3D(n.TEXTURE_2D_ARRAY, Re, be, ke[0].width, ke[0].height, le.depth);
          for (let ce = 0, se = ke.length; ce < se; ce++)
            if (ge = ke[ce], _.format !== xn)
              if (Me !== null)
                if (F) {
                  if (pe)
                    if (_.layerUpdates.size > 0) {
                      const Ie = oh(ge.width, ge.height, _.format, _.type);
                      for (const Ge of _.layerUpdates) {
                        const ht = ge.data.subarray(
                          Ge * Ie / ge.data.BYTES_PER_ELEMENT,
                          (Ge + 1) * Ie / ge.data.BYTES_PER_ELEMENT
                        );
                        t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY, ce, 0, 0, Ge, ge.width, ge.height, 1, Me, ht);
                      }
                      _.clearLayerUpdates();
                    } else
                      t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY, ce, 0, 0, 0, ge.width, ge.height, le.depth, Me, ge.data);
                } else
                  t.compressedTexImage3D(n.TEXTURE_2D_ARRAY, ce, be, ge.width, ge.height, le.depth, 0, ge.data, 0, 0);
              else
                console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");
            else
              F ? pe && t.texSubImage3D(n.TEXTURE_2D_ARRAY, ce, 0, 0, 0, ge.width, ge.height, le.depth, Me, Ce, ge.data) : t.texImage3D(n.TEXTURE_2D_ARRAY, ce, be, ge.width, ge.height, le.depth, 0, Me, Ce, ge.data);
        } else {
          F && he && t.texStorage2D(n.TEXTURE_2D, Re, be, ke[0].width, ke[0].height);
          for (let ce = 0, se = ke.length; ce < se; ce++)
            ge = ke[ce], _.format !== xn ? Me !== null ? F ? pe && t.compressedTexSubImage2D(n.TEXTURE_2D, ce, 0, 0, ge.width, ge.height, Me, ge.data) : t.compressedTexImage2D(n.TEXTURE_2D, ce, be, ge.width, ge.height, 0, ge.data) : console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()") : F ? pe && t.texSubImage2D(n.TEXTURE_2D, ce, 0, 0, ge.width, ge.height, Me, Ce, ge.data) : t.texImage2D(n.TEXTURE_2D, ce, be, ge.width, ge.height, 0, Me, Ce, ge.data);
        }
      else if (_.isDataArrayTexture)
        if (F) {
          if (he && t.texStorage3D(n.TEXTURE_2D_ARRAY, Re, be, le.width, le.height, le.depth), pe)
            if (_.layerUpdates.size > 0) {
              const ce = oh(le.width, le.height, _.format, _.type);
              for (const se of _.layerUpdates) {
                const Ie = le.data.subarray(
                  se * ce / le.data.BYTES_PER_ELEMENT,
                  (se + 1) * ce / le.data.BYTES_PER_ELEMENT
                );
                t.texSubImage3D(n.TEXTURE_2D_ARRAY, 0, 0, 0, se, le.width, le.height, 1, Me, Ce, Ie);
              }
              _.clearLayerUpdates();
            } else
              t.texSubImage3D(n.TEXTURE_2D_ARRAY, 0, 0, 0, 0, le.width, le.height, le.depth, Me, Ce, le.data);
        } else
          t.texImage3D(n.TEXTURE_2D_ARRAY, 0, be, le.width, le.height, le.depth, 0, Me, Ce, le.data);
      else if (_.isData3DTexture)
        F ? (he && t.texStorage3D(n.TEXTURE_3D, Re, be, le.width, le.height, le.depth), pe && t.texSubImage3D(n.TEXTURE_3D, 0, 0, 0, 0, le.width, le.height, le.depth, Me, Ce, le.data)) : t.texImage3D(n.TEXTURE_3D, 0, be, le.width, le.height, le.depth, 0, Me, Ce, le.data);
      else if (_.isFramebufferTexture) {
        if (he)
          if (F)
            t.texStorage2D(n.TEXTURE_2D, Re, be, le.width, le.height);
          else {
            let ce = le.width, se = le.height;
            for (let Ie = 0; Ie < Re; Ie++)
              t.texImage2D(n.TEXTURE_2D, Ie, be, ce, se, 0, Me, Ce, null), ce >>= 1, se >>= 1;
          }
      } else if (ke.length > 0) {
        if (F && he) {
          const ce = Se(ke[0]);
          t.texStorage2D(n.TEXTURE_2D, Re, be, ce.width, ce.height);
        }
        for (let ce = 0, se = ke.length; ce < se; ce++)
          ge = ke[ce], F ? pe && t.texSubImage2D(n.TEXTURE_2D, ce, 0, 0, Me, Ce, ge) : t.texImage2D(n.TEXTURE_2D, ce, be, Me, Ce, ge);
        _.generateMipmaps = !1;
      } else if (F) {
        if (he) {
          const ce = Se(le);
          t.texStorage2D(n.TEXTURE_2D, Re, be, ce.width, ce.height);
        }
        pe && t.texSubImage2D(n.TEXTURE_2D, 0, 0, 0, Me, Ce, le);
      } else
        t.texImage2D(n.TEXTURE_2D, 0, be, Me, Ce, le);
      m(_) && d(k), _e.__version = G.version, _.onUpdate && _.onUpdate(_);
    }
    E.__version = _.version;
  }
  function re(E, _, I) {
    if (_.image.length !== 6) return;
    const k = Le(E, _), J = _.source;
    t.bindTexture(n.TEXTURE_CUBE_MAP, E.__webglTexture, n.TEXTURE0 + I);
    const G = i.get(J);
    if (J.version !== G.__version || k === !0) {
      t.activeTexture(n.TEXTURE0 + I);
      const _e = Qe.getPrimaries(Qe.workingColorSpace), oe = _.colorSpace === gi ? null : Qe.getPrimaries(_.colorSpace), Ee = _.colorSpace === gi || _e === oe ? n.NONE : n.BROWSER_DEFAULT_WEBGL;
      n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL, _.flipY), n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL, _.premultiplyAlpha), n.pixelStorei(n.UNPACK_ALIGNMENT, _.unpackAlignment), n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL, Ee);
      const Te = _.isCompressedTexture || _.image[0].isCompressedTexture, le = _.image[0] && _.image[0].isDataTexture, Me = [];
      for (let se = 0; se < 6; se++)
        !Te && !le ? Me[se] = x(_.image[se], !0, s.maxCubemapSize) : Me[se] = le ? _.image[se].image : _.image[se], Me[se] = ee(_, Me[se]);
      const Ce = Me[0], be = r.convert(_.format, _.colorSpace), ge = r.convert(_.type), ke = A(_.internalFormat, be, ge, _.colorSpace), F = _.isVideoTexture !== !0, he = G.__version === void 0 || k === !0, pe = J.dataReady;
      let Re = R(_, Ce);
      de(n.TEXTURE_CUBE_MAP, _);
      let ce;
      if (Te) {
        F && he && t.texStorage2D(n.TEXTURE_CUBE_MAP, Re, ke, Ce.width, Ce.height);
        for (let se = 0; se < 6; se++) {
          ce = Me[se].mipmaps;
          for (let Ie = 0; Ie < ce.length; Ie++) {
            const Ge = ce[Ie];
            _.format !== xn ? be !== null ? F ? pe && t.compressedTexSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Ie, 0, 0, Ge.width, Ge.height, be, Ge.data) : t.compressedTexImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Ie, ke, Ge.width, Ge.height, 0, Ge.data) : console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()") : F ? pe && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Ie, 0, 0, Ge.width, Ge.height, be, ge, Ge.data) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Ie, ke, Ge.width, Ge.height, 0, be, ge, Ge.data);
          }
        }
      } else {
        if (ce = _.mipmaps, F && he) {
          ce.length > 0 && Re++;
          const se = Se(Me[0]);
          t.texStorage2D(n.TEXTURE_CUBE_MAP, Re, ke, se.width, se.height);
        }
        for (let se = 0; se < 6; se++)
          if (le) {
            F ? pe && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, 0, 0, 0, Me[se].width, Me[se].height, be, ge, Me[se].data) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, 0, ke, Me[se].width, Me[se].height, 0, be, ge, Me[se].data);
            for (let Ie = 0; Ie < ce.length; Ie++) {
              const ht = ce[Ie].image[se].image;
              F ? pe && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Ie + 1, 0, 0, ht.width, ht.height, be, ge, ht.data) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Ie + 1, ke, ht.width, ht.height, 0, be, ge, ht.data);
            }
          } else {
            F ? pe && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, 0, 0, 0, be, ge, Me[se]) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, 0, ke, be, ge, Me[se]);
            for (let Ie = 0; Ie < ce.length; Ie++) {
              const Ge = ce[Ie];
              F ? pe && t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Ie + 1, 0, 0, be, ge, Ge.image[se]) : t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X + se, Ie + 1, ke, be, ge, Ge.image[se]);
            }
          }
      }
      m(_) && d(n.TEXTURE_CUBE_MAP), G.__version = J.version, _.onUpdate && _.onUpdate(_);
    }
    E.__version = _.version;
  }
  function Ae(E, _, I, k, J, G) {
    const _e = r.convert(I.format, I.colorSpace), oe = r.convert(I.type), Ee = A(I.internalFormat, _e, oe, I.colorSpace), Te = i.get(_), le = i.get(I);
    if (le.__renderTarget = _, !Te.__hasExternalTextures) {
      const Me = Math.max(1, _.width >> G), Ce = Math.max(1, _.height >> G);
      J === n.TEXTURE_3D || J === n.TEXTURE_2D_ARRAY ? t.texImage3D(J, G, Ee, Me, Ce, _.depth, 0, _e, oe, null) : t.texImage2D(J, G, Ee, Me, Ce, 0, _e, oe, null);
    }
    t.bindFramebuffer(n.FRAMEBUFFER, E), q(_) ? a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER, k, J, le.__webglTexture, 0, ae(_)) : (J === n.TEXTURE_2D || J >= n.TEXTURE_CUBE_MAP_POSITIVE_X && J <= n.TEXTURE_CUBE_MAP_NEGATIVE_Z) && n.framebufferTexture2D(n.FRAMEBUFFER, k, J, le.__webglTexture, G), t.bindFramebuffer(n.FRAMEBUFFER, null);
  }
  function Oe(E, _, I) {
    if (n.bindRenderbuffer(n.RENDERBUFFER, E), _.depthBuffer) {
      const k = _.depthTexture, J = k && k.isDepthTexture ? k.type : null, G = M(_.stencilBuffer, J), _e = _.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT, oe = ae(_);
      q(_) ? a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER, oe, G, _.width, _.height) : I ? n.renderbufferStorageMultisample(n.RENDERBUFFER, oe, G, _.width, _.height) : n.renderbufferStorage(n.RENDERBUFFER, G, _.width, _.height), n.framebufferRenderbuffer(n.FRAMEBUFFER, _e, n.RENDERBUFFER, E);
    } else {
      const k = _.textures;
      for (let J = 0; J < k.length; J++) {
        const G = k[J], _e = r.convert(G.format, G.colorSpace), oe = r.convert(G.type), Ee = A(G.internalFormat, _e, oe, G.colorSpace), Te = ae(_);
        I && q(_) === !1 ? n.renderbufferStorageMultisample(n.RENDERBUFFER, Te, Ee, _.width, _.height) : q(_) ? a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER, Te, Ee, _.width, _.height) : n.renderbufferStorage(n.RENDERBUFFER, Ee, _.width, _.height);
      }
    }
    n.bindRenderbuffer(n.RENDERBUFFER, null);
  }
  function Pe(E, _) {
    if (_ && _.isWebGLCubeRenderTarget) throw new Error("Depth Texture with cube render targets is not supported");
    if (t.bindFramebuffer(n.FRAMEBUFFER, E), !(_.depthTexture && _.depthTexture.isDepthTexture))
      throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");
    const k = i.get(_.depthTexture);
    k.__renderTarget = _, (!k.__webglTexture || _.depthTexture.image.width !== _.width || _.depthTexture.image.height !== _.height) && (_.depthTexture.image.width = _.width, _.depthTexture.image.height = _.height, _.depthTexture.needsUpdate = !0), te(_.depthTexture, 0);
    const J = k.__webglTexture, G = ae(_);
    if (_.depthTexture.format === Ar)
      q(_) ? a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER, n.DEPTH_ATTACHMENT, n.TEXTURE_2D, J, 0, G) : n.framebufferTexture2D(n.FRAMEBUFFER, n.DEPTH_ATTACHMENT, n.TEXTURE_2D, J, 0);
    else if (_.depthTexture.format === wr)
      q(_) ? a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER, n.DEPTH_STENCIL_ATTACHMENT, n.TEXTURE_2D, J, 0, G) : n.framebufferTexture2D(n.FRAMEBUFFER, n.DEPTH_STENCIL_ATTACHMENT, n.TEXTURE_2D, J, 0);
    else
      throw new Error("Unknown depthTexture format");
  }
  function $e(E) {
    const _ = i.get(E), I = E.isWebGLCubeRenderTarget === !0;
    if (_.__boundDepthTexture !== E.depthTexture) {
      const k = E.depthTexture;
      if (_.__depthDisposeCallback && _.__depthDisposeCallback(), k) {
        const J = () => {
          delete _.__boundDepthTexture, delete _.__depthDisposeCallback, k.removeEventListener("dispose", J);
        };
        k.addEventListener("dispose", J), _.__depthDisposeCallback = J;
      }
      _.__boundDepthTexture = k;
    }
    if (E.depthTexture && !_.__autoAllocateDepthBuffer) {
      if (I) throw new Error("target.depthTexture not supported in Cube render targets");
      const k = E.texture.mipmaps;
      k && k.length > 0 ? Pe(_.__webglFramebuffer[0], E) : Pe(_.__webglFramebuffer, E);
    } else if (I) {
      _.__webglDepthbuffer = [];
      for (let k = 0; k < 6; k++)
        if (t.bindFramebuffer(n.FRAMEBUFFER, _.__webglFramebuffer[k]), _.__webglDepthbuffer[k] === void 0)
          _.__webglDepthbuffer[k] = n.createRenderbuffer(), Oe(_.__webglDepthbuffer[k], E, !1);
        else {
          const J = E.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT, G = _.__webglDepthbuffer[k];
          n.bindRenderbuffer(n.RENDERBUFFER, G), n.framebufferRenderbuffer(n.FRAMEBUFFER, J, n.RENDERBUFFER, G);
        }
    } else {
      const k = E.texture.mipmaps;
      if (k && k.length > 0 ? t.bindFramebuffer(n.FRAMEBUFFER, _.__webglFramebuffer[0]) : t.bindFramebuffer(n.FRAMEBUFFER, _.__webglFramebuffer), _.__webglDepthbuffer === void 0)
        _.__webglDepthbuffer = n.createRenderbuffer(), Oe(_.__webglDepthbuffer, E, !1);
      else {
        const J = E.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT, G = _.__webglDepthbuffer;
        n.bindRenderbuffer(n.RENDERBUFFER, G), n.framebufferRenderbuffer(n.FRAMEBUFFER, J, n.RENDERBUFFER, G);
      }
    }
    t.bindFramebuffer(n.FRAMEBUFFER, null);
  }
  function C(E, _, I) {
    const k = i.get(E);
    _ !== void 0 && Ae(k.__webglFramebuffer, E, E.texture, n.COLOR_ATTACHMENT0, n.TEXTURE_2D, 0), I !== void 0 && $e(E);
  }
  function g(E) {
    const _ = E.texture, I = i.get(E), k = i.get(_);
    E.addEventListener("dispose", D);
    const J = E.textures, G = E.isWebGLCubeRenderTarget === !0, _e = J.length > 1;
    if (_e || (k.__webglTexture === void 0 && (k.__webglTexture = n.createTexture()), k.__version = _.version, o.memory.textures++), G) {
      I.__webglFramebuffer = [];
      for (let oe = 0; oe < 6; oe++)
        if (_.mipmaps && _.mipmaps.length > 0) {
          I.__webglFramebuffer[oe] = [];
          for (let Ee = 0; Ee < _.mipmaps.length; Ee++)
            I.__webglFramebuffer[oe][Ee] = n.createFramebuffer();
        } else
          I.__webglFramebuffer[oe] = n.createFramebuffer();
    } else {
      if (_.mipmaps && _.mipmaps.length > 0) {
        I.__webglFramebuffer = [];
        for (let oe = 0; oe < _.mipmaps.length; oe++)
          I.__webglFramebuffer[oe] = n.createFramebuffer();
      } else
        I.__webglFramebuffer = n.createFramebuffer();
      if (_e)
        for (let oe = 0, Ee = J.length; oe < Ee; oe++) {
          const Te = i.get(J[oe]);
          Te.__webglTexture === void 0 && (Te.__webglTexture = n.createTexture(), o.memory.textures++);
        }
      if (E.samples > 0 && q(E) === !1) {
        I.__webglMultisampledFramebuffer = n.createFramebuffer(), I.__webglColorRenderbuffer = [], t.bindFramebuffer(n.FRAMEBUFFER, I.__webglMultisampledFramebuffer);
        for (let oe = 0; oe < J.length; oe++) {
          const Ee = J[oe];
          I.__webglColorRenderbuffer[oe] = n.createRenderbuffer(), n.bindRenderbuffer(n.RENDERBUFFER, I.__webglColorRenderbuffer[oe]);
          const Te = r.convert(Ee.format, Ee.colorSpace), le = r.convert(Ee.type), Me = A(Ee.internalFormat, Te, le, Ee.colorSpace, E.isXRRenderTarget === !0), Ce = ae(E);
          n.renderbufferStorageMultisample(n.RENDERBUFFER, Ce, Me, E.width, E.height), n.framebufferRenderbuffer(n.FRAMEBUFFER, n.COLOR_ATTACHMENT0 + oe, n.RENDERBUFFER, I.__webglColorRenderbuffer[oe]);
        }
        n.bindRenderbuffer(n.RENDERBUFFER, null), E.depthBuffer && (I.__webglDepthRenderbuffer = n.createRenderbuffer(), Oe(I.__webglDepthRenderbuffer, E, !0)), t.bindFramebuffer(n.FRAMEBUFFER, null);
      }
    }
    if (G) {
      t.bindTexture(n.TEXTURE_CUBE_MAP, k.__webglTexture), de(n.TEXTURE_CUBE_MAP, _);
      for (let oe = 0; oe < 6; oe++)
        if (_.mipmaps && _.mipmaps.length > 0)
          for (let Ee = 0; Ee < _.mipmaps.length; Ee++)
            Ae(I.__webglFramebuffer[oe][Ee], E, _, n.COLOR_ATTACHMENT0, n.TEXTURE_CUBE_MAP_POSITIVE_X + oe, Ee);
        else
          Ae(I.__webglFramebuffer[oe], E, _, n.COLOR_ATTACHMENT0, n.TEXTURE_CUBE_MAP_POSITIVE_X + oe, 0);
      m(_) && d(n.TEXTURE_CUBE_MAP), t.unbindTexture();
    } else if (_e) {
      for (let oe = 0, Ee = J.length; oe < Ee; oe++) {
        const Te = J[oe], le = i.get(Te);
        let Me = n.TEXTURE_2D;
        (E.isWebGL3DRenderTarget || E.isWebGLArrayRenderTarget) && (Me = E.isWebGL3DRenderTarget ? n.TEXTURE_3D : n.TEXTURE_2D_ARRAY), t.bindTexture(Me, le.__webglTexture), de(Me, Te), Ae(I.__webglFramebuffer, E, Te, n.COLOR_ATTACHMENT0 + oe, Me, 0), m(Te) && d(Me);
      }
      t.unbindTexture();
    } else {
      let oe = n.TEXTURE_2D;
      if ((E.isWebGL3DRenderTarget || E.isWebGLArrayRenderTarget) && (oe = E.isWebGL3DRenderTarget ? n.TEXTURE_3D : n.TEXTURE_2D_ARRAY), t.bindTexture(oe, k.__webglTexture), de(oe, _), _.mipmaps && _.mipmaps.length > 0)
        for (let Ee = 0; Ee < _.mipmaps.length; Ee++)
          Ae(I.__webglFramebuffer[Ee], E, _, n.COLOR_ATTACHMENT0, oe, Ee);
      else
        Ae(I.__webglFramebuffer, E, _, n.COLOR_ATTACHMENT0, oe, 0);
      m(_) && d(oe), t.unbindTexture();
    }
    E.depthBuffer && $e(E);
  }
  function W(E) {
    const _ = E.textures;
    for (let I = 0, k = _.length; I < k; I++) {
      const J = _[I];
      if (m(J)) {
        const G = b(E), _e = i.get(J).__webglTexture;
        t.bindTexture(G, _e), d(G), t.unbindTexture();
      }
    }
  }
  const j = [], X = [];
  function z(E) {
    if (E.samples > 0) {
      if (q(E) === !1) {
        const _ = E.textures, I = E.width, k = E.height;
        let J = n.COLOR_BUFFER_BIT;
        const G = E.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT, _e = i.get(E), oe = _.length > 1;
        if (oe)
          for (let Te = 0; Te < _.length; Te++)
            t.bindFramebuffer(n.FRAMEBUFFER, _e.__webglMultisampledFramebuffer), n.framebufferRenderbuffer(n.FRAMEBUFFER, n.COLOR_ATTACHMENT0 + Te, n.RENDERBUFFER, null), t.bindFramebuffer(n.FRAMEBUFFER, _e.__webglFramebuffer), n.framebufferTexture2D(n.DRAW_FRAMEBUFFER, n.COLOR_ATTACHMENT0 + Te, n.TEXTURE_2D, null, 0);
        t.bindFramebuffer(n.READ_FRAMEBUFFER, _e.__webglMultisampledFramebuffer);
        const Ee = E.texture.mipmaps;
        Ee && Ee.length > 0 ? t.bindFramebuffer(n.DRAW_FRAMEBUFFER, _e.__webglFramebuffer[0]) : t.bindFramebuffer(n.DRAW_FRAMEBUFFER, _e.__webglFramebuffer);
        for (let Te = 0; Te < _.length; Te++) {
          if (E.resolveDepthBuffer && (E.depthBuffer && (J |= n.DEPTH_BUFFER_BIT), E.stencilBuffer && E.resolveStencilBuffer && (J |= n.STENCIL_BUFFER_BIT)), oe) {
            n.framebufferRenderbuffer(n.READ_FRAMEBUFFER, n.COLOR_ATTACHMENT0, n.RENDERBUFFER, _e.__webglColorRenderbuffer[Te]);
            const le = i.get(_[Te]).__webglTexture;
            n.framebufferTexture2D(n.DRAW_FRAMEBUFFER, n.COLOR_ATTACHMENT0, n.TEXTURE_2D, le, 0);
          }
          n.blitFramebuffer(0, 0, I, k, 0, 0, I, k, J, n.NEAREST), l === !0 && (j.length = 0, X.length = 0, j.push(n.COLOR_ATTACHMENT0 + Te), E.depthBuffer && E.resolveDepthBuffer === !1 && (j.push(G), X.push(G), n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER, X)), n.invalidateFramebuffer(n.READ_FRAMEBUFFER, j));
        }
        if (t.bindFramebuffer(n.READ_FRAMEBUFFER, null), t.bindFramebuffer(n.DRAW_FRAMEBUFFER, null), oe)
          for (let Te = 0; Te < _.length; Te++) {
            t.bindFramebuffer(n.FRAMEBUFFER, _e.__webglMultisampledFramebuffer), n.framebufferRenderbuffer(n.FRAMEBUFFER, n.COLOR_ATTACHMENT0 + Te, n.RENDERBUFFER, _e.__webglColorRenderbuffer[Te]);
            const le = i.get(_[Te]).__webglTexture;
            t.bindFramebuffer(n.FRAMEBUFFER, _e.__webglFramebuffer), n.framebufferTexture2D(n.DRAW_FRAMEBUFFER, n.COLOR_ATTACHMENT0 + Te, n.TEXTURE_2D, le, 0);
          }
        t.bindFramebuffer(n.DRAW_FRAMEBUFFER, _e.__webglMultisampledFramebuffer);
      } else if (E.depthBuffer && E.resolveDepthBuffer === !1 && l) {
        const _ = E.stencilBuffer ? n.DEPTH_STENCIL_ATTACHMENT : n.DEPTH_ATTACHMENT;
        n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER, [_]);
      }
    }
  }
  function ae(E) {
    return Math.min(s.maxSamples, E.samples);
  }
  function q(E) {
    const _ = i.get(E);
    return E.samples > 0 && e.has("WEBGL_multisampled_render_to_texture") === !0 && _.__useRenderToTexture !== !1;
  }
  function Q(E) {
    const _ = o.render.frame;
    u.get(E) !== _ && (u.set(E, _), E.update());
  }
  function ee(E, _) {
    const I = E.colorSpace, k = E.format, J = E.type;
    return E.isCompressedTexture === !0 || E.isVideoTexture === !0 || I !== Os && I !== gi && (Qe.getTransfer(I) === ot ? (k !== xn || J !== Bn) && console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType.") : console.error("THREE.WebGLTextures: Unsupported texture color space:", I)), _;
  }
  function Se(E) {
    return typeof HTMLImageElement < "u" && E instanceof HTMLImageElement ? (c.width = E.naturalWidth || E.width, c.height = E.naturalHeight || E.height) : typeof VideoFrame < "u" && E instanceof VideoFrame ? (c.width = E.displayWidth, c.height = E.displayHeight) : (c.width = E.width, c.height = E.height), c;
  }
  this.allocateTextureUnit = V, this.resetTextureUnits = L, this.setTexture2D = te, this.setTexture2DArray = $, this.setTexture3D = ie, this.setTextureCube = H, this.rebindTextures = C, this.setupRenderTarget = g, this.updateRenderTargetMipmap = W, this.updateMultisampleRenderTarget = z, this.setupDepthRenderbuffer = $e, this.setupFrameBufferTexture = Ae, this.useMultisampledRTT = q;
}
function YS(n, e) {
  function t(i, s = gi) {
    let r;
    const o = Qe.getTransfer(s);
    if (i === Bn) return n.UNSIGNED_BYTE;
    if (i === Mc) return n.UNSIGNED_SHORT_4_4_4_4;
    if (i === Sc) return n.UNSIGNED_SHORT_5_5_5_1;
    if (i === sd) return n.UNSIGNED_INT_5_9_9_9_REV;
    if (i === rd) return n.UNSIGNED_INT_10F_11F_11F_REV;
    if (i === nd) return n.BYTE;
    if (i === id) return n.SHORT;
    if (i === Tr) return n.UNSIGNED_SHORT;
    if (i === xc) return n.INT;
    if (i === Xi) return n.UNSIGNED_INT;
    if (i === ei) return n.FLOAT;
    if (i === Lr) return n.HALF_FLOAT;
    if (i === od) return n.ALPHA;
    if (i === ad) return n.RGB;
    if (i === xn) return n.RGBA;
    if (i === Ar) return n.DEPTH_COMPONENT;
    if (i === wr) return n.DEPTH_STENCIL;
    if (i === ld) return n.RED;
    if (i === yc) return n.RED_INTEGER;
    if (i === cd) return n.RG;
    if (i === Ec) return n.RG_INTEGER;
    if (i === Tc) return n.RGBA_INTEGER;
    if (i === So || i === yo || i === Eo || i === To)
      if (o === ot)
        if (r = e.get("WEBGL_compressed_texture_s3tc_srgb"), r !== null) {
          if (i === So) return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;
          if (i === yo) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
          if (i === Eo) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
          if (i === To) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
        } else
          return null;
      else if (r = e.get("WEBGL_compressed_texture_s3tc"), r !== null) {
        if (i === So) return r.COMPRESSED_RGB_S3TC_DXT1_EXT;
        if (i === yo) return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;
        if (i === Eo) return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;
        if (i === To) return r.COMPRESSED_RGBA_S3TC_DXT5_EXT;
      } else
        return null;
    if (i === bl || i === Al || i === wl || i === Rl)
      if (r = e.get("WEBGL_compressed_texture_pvrtc"), r !== null) {
        if (i === bl) return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
        if (i === Al) return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
        if (i === wl) return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
        if (i === Rl) return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
      } else
        return null;
    if (i === Cl || i === Pl || i === Dl)
      if (r = e.get("WEBGL_compressed_texture_etc"), r !== null) {
        if (i === Cl || i === Pl) return o === ot ? r.COMPRESSED_SRGB8_ETC2 : r.COMPRESSED_RGB8_ETC2;
        if (i === Dl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC : r.COMPRESSED_RGBA8_ETC2_EAC;
      } else
        return null;
    if (i === Ll || i === Il || i === Ul || i === Nl || i === Fl || i === Ol || i === Bl || i === zl || i === Hl || i === Vl || i === kl || i === Gl || i === Wl || i === Xl)
      if (r = e.get("WEBGL_compressed_texture_astc"), r !== null) {
        if (i === Ll) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR : r.COMPRESSED_RGBA_ASTC_4x4_KHR;
        if (i === Il) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR : r.COMPRESSED_RGBA_ASTC_5x4_KHR;
        if (i === Ul) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR : r.COMPRESSED_RGBA_ASTC_5x5_KHR;
        if (i === Nl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR : r.COMPRESSED_RGBA_ASTC_6x5_KHR;
        if (i === Fl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR : r.COMPRESSED_RGBA_ASTC_6x6_KHR;
        if (i === Ol) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR : r.COMPRESSED_RGBA_ASTC_8x5_KHR;
        if (i === Bl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR : r.COMPRESSED_RGBA_ASTC_8x6_KHR;
        if (i === zl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR : r.COMPRESSED_RGBA_ASTC_8x8_KHR;
        if (i === Hl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR : r.COMPRESSED_RGBA_ASTC_10x5_KHR;
        if (i === Vl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR : r.COMPRESSED_RGBA_ASTC_10x6_KHR;
        if (i === kl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR : r.COMPRESSED_RGBA_ASTC_10x8_KHR;
        if (i === Gl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR : r.COMPRESSED_RGBA_ASTC_10x10_KHR;
        if (i === Wl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR : r.COMPRESSED_RGBA_ASTC_12x10_KHR;
        if (i === Xl) return o === ot ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR : r.COMPRESSED_RGBA_ASTC_12x12_KHR;
      } else
        return null;
    if (i === Yl || i === ql || i === jl)
      if (r = e.get("EXT_texture_compression_bptc"), r !== null) {
        if (i === Yl) return o === ot ? r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT : r.COMPRESSED_RGBA_BPTC_UNORM_EXT;
        if (i === ql) return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;
        if (i === jl) return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT;
      } else
        return null;
    if (i === Kl || i === $l || i === Zl || i === Jl)
      if (r = e.get("EXT_texture_compression_rgtc"), r !== null) {
        if (i === Kl) return r.COMPRESSED_RED_RGTC1_EXT;
        if (i === $l) return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;
        if (i === Zl) return r.COMPRESSED_RED_GREEN_RGTC2_EXT;
        if (i === Jl) return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
      } else
        return null;
    return i === br ? n.UNSIGNED_INT_24_8 : n[i] !== void 0 ? n[i] : null;
  }
  return { convert: t };
}
const qS = `
void main() {

	gl_Position = vec4( position, 1.0 );

}`, jS = `
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;
class KS {
  /**
   * Constructs a new depth sensing module.
   */
  constructor() {
    this.texture = null, this.mesh = null, this.depthNear = 0, this.depthFar = 0;
  }
  /**
   * Inits the depth sensing module
   *
   * @param {XRWebGLDepthInformation} depthData - The XR depth data.
   * @param {XRRenderState} renderState - The XR render state.
   */
  init(e, t) {
    if (this.texture === null) {
      const i = new Td(e.texture);
      (e.depthNear !== t.depthNear || e.depthFar !== t.depthFar) && (this.depthNear = e.depthNear, this.depthFar = e.depthFar), this.texture = i;
    }
  }
  /**
   * Returns a plane mesh that visualizes the depth texture.
   *
   * @param {ArrayCamera} cameraXR - The XR camera.
   * @return {?Mesh} The plane mesh.
   */
  getMesh(e) {
    if (this.texture !== null && this.mesh === null) {
      const t = e.cameras[0].viewport, i = new yi({
        vertexShader: qS,
        fragmentShader: jS,
        uniforms: {
          depthColor: { value: this.texture },
          depthWidth: { value: t.z },
          depthHeight: { value: t.w }
        }
      });
      this.mesh = new vt(new zs(20, 20), i);
    }
    return this.mesh;
  }
  /**
   * Resets the module
   */
  reset() {
    this.texture = null, this.mesh = null;
  }
  /**
   * Returns a texture representing the depth of the user's environment.
   *
   * @return {?ExternalTexture} The depth texture.
   */
  getDepthTexture() {
    return this.texture;
  }
}
class $S extends Zi {
  /**
   * Constructs a new WebGL renderer.
   *
   * @param {WebGLRenderer} renderer - The renderer.
   * @param {WebGL2RenderingContext} gl - The rendering context.
   */
  constructor(e, t) {
    super();
    const i = this;
    let s = null, r = 1, o = null, a = "local-floor", l = 1, c = null, u = null, h = null, f = null, p = null, v = null;
    const x = typeof XRWebGLBinding < "u", m = new KS(), d = {}, b = t.getContextAttributes();
    let A = null, M = null;
    const R = [], w = [], D = new Ve();
    let U = null;
    const y = new nn();
    y.viewport = new lt();
    const S = new nn();
    S.viewport = new lt();
    const P = [y, S], L = new m0();
    let V = null, Z = null;
    this.cameraAutoUpdate = !0, this.enabled = !1, this.isPresenting = !1, this.getController = function(ne) {
      let re = R[ne];
      return re === void 0 && (re = new Va(), R[ne] = re), re.getTargetRaySpace();
    }, this.getControllerGrip = function(ne) {
      let re = R[ne];
      return re === void 0 && (re = new Va(), R[ne] = re), re.getGripSpace();
    }, this.getHand = function(ne) {
      let re = R[ne];
      return re === void 0 && (re = new Va(), R[ne] = re), re.getHandSpace();
    };
    function te(ne) {
      const re = w.indexOf(ne.inputSource);
      if (re === -1)
        return;
      const Ae = R[re];
      Ae !== void 0 && (Ae.update(ne.inputSource, ne.frame, c || o), Ae.dispatchEvent({ type: ne.type, data: ne.inputSource }));
    }
    function $() {
      s.removeEventListener("select", te), s.removeEventListener("selectstart", te), s.removeEventListener("selectend", te), s.removeEventListener("squeeze", te), s.removeEventListener("squeezestart", te), s.removeEventListener("squeezeend", te), s.removeEventListener("end", $), s.removeEventListener("inputsourceschange", ie);
      for (let ne = 0; ne < R.length; ne++) {
        const re = w[ne];
        re !== null && (w[ne] = null, R[ne].disconnect(re));
      }
      V = null, Z = null, m.reset();
      for (const ne in d)
        delete d[ne];
      e.setRenderTarget(A), p = null, f = null, h = null, s = null, M = null, Ze.stop(), i.isPresenting = !1, e.setPixelRatio(U), e.setSize(D.width, D.height, !1), i.dispatchEvent({ type: "sessionend" });
    }
    this.setFramebufferScaleFactor = function(ne) {
      r = ne, i.isPresenting === !0 && console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.");
    }, this.setReferenceSpaceType = function(ne) {
      a = ne, i.isPresenting === !0 && console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.");
    }, this.getReferenceSpace = function() {
      return c || o;
    }, this.setReferenceSpace = function(ne) {
      c = ne;
    }, this.getBaseLayer = function() {
      return f !== null ? f : p;
    }, this.getBinding = function() {
      return h === null && x && (h = new XRWebGLBinding(s, t)), h;
    }, this.getFrame = function() {
      return v;
    }, this.getSession = function() {
      return s;
    }, this.setSession = async function(ne) {
      if (s = ne, s !== null) {
        if (A = e.getRenderTarget(), s.addEventListener("select", te), s.addEventListener("selectstart", te), s.addEventListener("selectend", te), s.addEventListener("squeeze", te), s.addEventListener("squeezestart", te), s.addEventListener("squeezeend", te), s.addEventListener("end", $), s.addEventListener("inputsourceschange", ie), b.xrCompatible !== !0 && await t.makeXRCompatible(), U = e.getPixelRatio(), e.getSize(D), x && "createProjectionLayer" in XRWebGLBinding.prototype) {
          let Ae = null, Oe = null, Pe = null;
          b.depth && (Pe = b.stencil ? t.DEPTH24_STENCIL8 : t.DEPTH_COMPONENT24, Ae = b.stencil ? wr : Ar, Oe = b.stencil ? br : Xi);
          const $e = {
            colorFormat: t.RGBA8,
            depthFormat: Pe,
            scaleFactor: r
          };
          h = this.getBinding(), f = h.createProjectionLayer($e), s.updateRenderState({ layers: [f] }), e.setPixelRatio(1), e.setSize(f.textureWidth, f.textureHeight, !1), M = new qi(
            f.textureWidth,
            f.textureHeight,
            {
              format: xn,
              type: Bn,
              depthTexture: new Ed(f.textureWidth, f.textureHeight, Oe, void 0, void 0, void 0, void 0, void 0, void 0, Ae),
              stencilBuffer: b.stencil,
              colorSpace: e.outputColorSpace,
              samples: b.antialias ? 4 : 0,
              resolveDepthBuffer: f.ignoreDepthValues === !1,
              resolveStencilBuffer: f.ignoreDepthValues === !1
            }
          );
        } else {
          const Ae = {
            antialias: b.antialias,
            alpha: !0,
            depth: b.depth,
            stencil: b.stencil,
            framebufferScaleFactor: r
          };
          p = new XRWebGLLayer(s, t, Ae), s.updateRenderState({ baseLayer: p }), e.setPixelRatio(1), e.setSize(p.framebufferWidth, p.framebufferHeight, !1), M = new qi(
            p.framebufferWidth,
            p.framebufferHeight,
            {
              format: xn,
              type: Bn,
              colorSpace: e.outputColorSpace,
              stencilBuffer: b.stencil,
              resolveDepthBuffer: p.ignoreDepthValues === !1,
              resolveStencilBuffer: p.ignoreDepthValues === !1
            }
          );
        }
        M.isXRRenderTarget = !0, this.setFoveation(l), c = null, o = await s.requestReferenceSpace(a), Ze.setContext(s), Ze.start(), i.isPresenting = !0, i.dispatchEvent({ type: "sessionstart" });
      }
    }, this.getEnvironmentBlendMode = function() {
      if (s !== null)
        return s.environmentBlendMode;
    }, this.getDepthTexture = function() {
      return m.getDepthTexture();
    };
    function ie(ne) {
      for (let re = 0; re < ne.removed.length; re++) {
        const Ae = ne.removed[re], Oe = w.indexOf(Ae);
        Oe >= 0 && (w[Oe] = null, R[Oe].disconnect(Ae));
      }
      for (let re = 0; re < ne.added.length; re++) {
        const Ae = ne.added[re];
        let Oe = w.indexOf(Ae);
        if (Oe === -1) {
          for (let $e = 0; $e < R.length; $e++)
            if ($e >= w.length) {
              w.push(Ae), Oe = $e;
              break;
            } else if (w[$e] === null) {
              w[$e] = Ae, Oe = $e;
              break;
            }
          if (Oe === -1) break;
        }
        const Pe = R[Oe];
        Pe && Pe.connect(Ae);
      }
    }
    const H = new N(), fe = new N();
    function xe(ne, re, Ae) {
      H.setFromMatrixPosition(re.matrixWorld), fe.setFromMatrixPosition(Ae.matrixWorld);
      const Oe = H.distanceTo(fe), Pe = re.projectionMatrix.elements, $e = Ae.projectionMatrix.elements, C = Pe[14] / (Pe[10] - 1), g = Pe[14] / (Pe[10] + 1), W = (Pe[9] + 1) / Pe[5], j = (Pe[9] - 1) / Pe[5], X = (Pe[8] - 1) / Pe[0], z = ($e[8] + 1) / $e[0], ae = C * X, q = C * z, Q = Oe / (-X + z), ee = Q * -X;
      if (re.matrixWorld.decompose(ne.position, ne.quaternion, ne.scale), ne.translateX(ee), ne.translateZ(Q), ne.matrixWorld.compose(ne.position, ne.quaternion, ne.scale), ne.matrixWorldInverse.copy(ne.matrixWorld).invert(), Pe[10] === -1)
        ne.projectionMatrix.copy(re.projectionMatrix), ne.projectionMatrixInverse.copy(re.projectionMatrixInverse);
      else {
        const Se = C + Q, E = g + Q, _ = ae - ee, I = q + (Oe - ee), k = W * g / E * Se, J = j * g / E * Se;
        ne.projectionMatrix.makePerspective(_, I, k, J, Se, E), ne.projectionMatrixInverse.copy(ne.projectionMatrix).invert();
      }
    }
    function me(ne, re) {
      re === null ? ne.matrixWorld.copy(ne.matrix) : ne.matrixWorld.multiplyMatrices(re.matrixWorld, ne.matrix), ne.matrixWorldInverse.copy(ne.matrixWorld).invert();
    }
    this.updateCamera = function(ne) {
      if (s === null) return;
      let re = ne.near, Ae = ne.far;
      m.texture !== null && (m.depthNear > 0 && (re = m.depthNear), m.depthFar > 0 && (Ae = m.depthFar)), L.near = S.near = y.near = re, L.far = S.far = y.far = Ae, (V !== L.near || Z !== L.far) && (s.updateRenderState({
        depthNear: L.near,
        depthFar: L.far
      }), V = L.near, Z = L.far), L.layers.mask = ne.layers.mask | 6, y.layers.mask = L.layers.mask & 3, S.layers.mask = L.layers.mask & 5;
      const Oe = ne.parent, Pe = L.cameras;
      me(L, Oe);
      for (let $e = 0; $e < Pe.length; $e++)
        me(Pe[$e], Oe);
      Pe.length === 2 ? xe(L, y, S) : L.projectionMatrix.copy(y.projectionMatrix), de(ne, L, Oe);
    };
    function de(ne, re, Ae) {
      Ae === null ? ne.matrix.copy(re.matrixWorld) : (ne.matrix.copy(Ae.matrixWorld), ne.matrix.invert(), ne.matrix.multiply(re.matrixWorld)), ne.matrix.decompose(ne.position, ne.quaternion, ne.scale), ne.updateMatrixWorld(!0), ne.projectionMatrix.copy(re.projectionMatrix), ne.projectionMatrixInverse.copy(re.projectionMatrixInverse), ne.isPerspectiveCamera && (ne.fov = Ql * 2 * Math.atan(1 / ne.projectionMatrix.elements[5]), ne.zoom = 1);
    }
    this.getCamera = function() {
      return L;
    }, this.getFoveation = function() {
      if (!(f === null && p === null))
        return l;
    }, this.setFoveation = function(ne) {
      l = ne, f !== null && (f.fixedFoveation = ne), p !== null && p.fixedFoveation !== void 0 && (p.fixedFoveation = ne);
    }, this.hasDepthSensing = function() {
      return m.texture !== null;
    }, this.getDepthSensingMesh = function() {
      return m.getMesh(L);
    }, this.getCameraTexture = function(ne) {
      return d[ne];
    };
    let Le = null;
    function tt(ne, re) {
      if (u = re.getViewerPose(c || o), v = re, u !== null) {
        const Ae = u.views;
        p !== null && (e.setRenderTargetFramebuffer(M, p.framebuffer), e.setRenderTarget(M));
        let Oe = !1;
        Ae.length !== L.cameras.length && (L.cameras.length = 0, Oe = !0);
        for (let g = 0; g < Ae.length; g++) {
          const W = Ae[g];
          let j = null;
          if (p !== null)
            j = p.getViewport(W);
          else {
            const z = h.getViewSubImage(f, W);
            j = z.viewport, g === 0 && (e.setRenderTargetTextures(
              M,
              z.colorTexture,
              z.depthStencilTexture
            ), e.setRenderTarget(M));
          }
          let X = P[g];
          X === void 0 && (X = new nn(), X.layers.enable(g), X.viewport = new lt(), P[g] = X), X.matrix.fromArray(W.transform.matrix), X.matrix.decompose(X.position, X.quaternion, X.scale), X.projectionMatrix.fromArray(W.projectionMatrix), X.projectionMatrixInverse.copy(X.projectionMatrix).invert(), X.viewport.set(j.x, j.y, j.width, j.height), g === 0 && (L.matrix.copy(X.matrix), L.matrix.decompose(L.position, L.quaternion, L.scale)), Oe === !0 && L.cameras.push(X);
        }
        const Pe = s.enabledFeatures;
        if (Pe && Pe.includes("depth-sensing") && s.depthUsage == "gpu-optimized" && x) {
          h = i.getBinding();
          const g = h.getDepthInformation(Ae[0]);
          g && g.isValid && g.texture && m.init(g, s.renderState);
        }
        if (Pe && Pe.includes("camera-access") && x) {
          e.state.unbindTexture(), h = i.getBinding();
          for (let g = 0; g < Ae.length; g++) {
            const W = Ae[g].camera;
            if (W) {
              let j = d[W];
              j || (j = new Td(), d[W] = j);
              const X = h.getCameraImage(W);
              j.sourceTexture = X;
            }
          }
        }
      }
      for (let Ae = 0; Ae < R.length; Ae++) {
        const Oe = w[Ae], Pe = R[Ae];
        Oe !== null && Pe !== void 0 && Pe.update(Oe, re, c || o);
      }
      Le && Le(ne, re), re.detectedPlanes && i.dispatchEvent({ type: "planesdetected", data: re }), v = null;
    }
    const Ze = new wd();
    Ze.setAnimationLoop(tt), this.setAnimationLoop = function(ne) {
      Le = ne;
    }, this.dispose = function() {
    };
  }
}
const Ui = /* @__PURE__ */ new zn(), ZS = /* @__PURE__ */ new pt();
function JS(n, e) {
  function t(m, d) {
    m.matrixAutoUpdate === !0 && m.updateMatrix(), d.value.copy(m.matrix);
  }
  function i(m, d) {
    d.color.getRGB(m.fogColor.value, vd(n)), d.isFog ? (m.fogNear.value = d.near, m.fogFar.value = d.far) : d.isFogExp2 && (m.fogDensity.value = d.density);
  }
  function s(m, d, b, A, M) {
    d.isMeshBasicMaterial || d.isMeshLambertMaterial ? r(m, d) : d.isMeshToonMaterial ? (r(m, d), h(m, d)) : d.isMeshPhongMaterial ? (r(m, d), u(m, d)) : d.isMeshStandardMaterial ? (r(m, d), f(m, d), d.isMeshPhysicalMaterial && p(m, d, M)) : d.isMeshMatcapMaterial ? (r(m, d), v(m, d)) : d.isMeshDepthMaterial ? r(m, d) : d.isMeshDistanceMaterial ? (r(m, d), x(m, d)) : d.isMeshNormalMaterial ? r(m, d) : d.isLineBasicMaterial ? (o(m, d), d.isLineDashedMaterial && a(m, d)) : d.isPointsMaterial ? l(m, d, b, A) : d.isSpriteMaterial ? c(m, d) : d.isShadowMaterial ? (m.color.value.copy(d.color), m.opacity.value = d.opacity) : d.isShaderMaterial && (d.uniformsNeedUpdate = !1);
  }
  function r(m, d) {
    m.opacity.value = d.opacity, d.color && m.diffuse.value.copy(d.color), d.emissive && m.emissive.value.copy(d.emissive).multiplyScalar(d.emissiveIntensity), d.map && (m.map.value = d.map, t(d.map, m.mapTransform)), d.alphaMap && (m.alphaMap.value = d.alphaMap, t(d.alphaMap, m.alphaMapTransform)), d.bumpMap && (m.bumpMap.value = d.bumpMap, t(d.bumpMap, m.bumpMapTransform), m.bumpScale.value = d.bumpScale, d.side === Wt && (m.bumpScale.value *= -1)), d.normalMap && (m.normalMap.value = d.normalMap, t(d.normalMap, m.normalMapTransform), m.normalScale.value.copy(d.normalScale), d.side === Wt && m.normalScale.value.negate()), d.displacementMap && (m.displacementMap.value = d.displacementMap, t(d.displacementMap, m.displacementMapTransform), m.displacementScale.value = d.displacementScale, m.displacementBias.value = d.displacementBias), d.emissiveMap && (m.emissiveMap.value = d.emissiveMap, t(d.emissiveMap, m.emissiveMapTransform)), d.specularMap && (m.specularMap.value = d.specularMap, t(d.specularMap, m.specularMapTransform)), d.alphaTest > 0 && (m.alphaTest.value = d.alphaTest);
    const b = e.get(d), A = b.envMap, M = b.envMapRotation;
    A && (m.envMap.value = A, Ui.copy(M), Ui.x *= -1, Ui.y *= -1, Ui.z *= -1, A.isCubeTexture && A.isRenderTargetTexture === !1 && (Ui.y *= -1, Ui.z *= -1), m.envMapRotation.value.setFromMatrix4(ZS.makeRotationFromEuler(Ui)), m.flipEnvMap.value = A.isCubeTexture && A.isRenderTargetTexture === !1 ? -1 : 1, m.reflectivity.value = d.reflectivity, m.ior.value = d.ior, m.refractionRatio.value = d.refractionRatio), d.lightMap && (m.lightMap.value = d.lightMap, m.lightMapIntensity.value = d.lightMapIntensity, t(d.lightMap, m.lightMapTransform)), d.aoMap && (m.aoMap.value = d.aoMap, m.aoMapIntensity.value = d.aoMapIntensity, t(d.aoMap, m.aoMapTransform));
  }
  function o(m, d) {
    m.diffuse.value.copy(d.color), m.opacity.value = d.opacity, d.map && (m.map.value = d.map, t(d.map, m.mapTransform));
  }
  function a(m, d) {
    m.dashSize.value = d.dashSize, m.totalSize.value = d.dashSize + d.gapSize, m.scale.value = d.scale;
  }
  function l(m, d, b, A) {
    m.diffuse.value.copy(d.color), m.opacity.value = d.opacity, m.size.value = d.size * b, m.scale.value = A * 0.5, d.map && (m.map.value = d.map, t(d.map, m.uvTransform)), d.alphaMap && (m.alphaMap.value = d.alphaMap, t(d.alphaMap, m.alphaMapTransform)), d.alphaTest > 0 && (m.alphaTest.value = d.alphaTest);
  }
  function c(m, d) {
    m.diffuse.value.copy(d.color), m.opacity.value = d.opacity, m.rotation.value = d.rotation, d.map && (m.map.value = d.map, t(d.map, m.mapTransform)), d.alphaMap && (m.alphaMap.value = d.alphaMap, t(d.alphaMap, m.alphaMapTransform)), d.alphaTest > 0 && (m.alphaTest.value = d.alphaTest);
  }
  function u(m, d) {
    m.specular.value.copy(d.specular), m.shininess.value = Math.max(d.shininess, 1e-4);
  }
  function h(m, d) {
    d.gradientMap && (m.gradientMap.value = d.gradientMap);
  }
  function f(m, d) {
    m.metalness.value = d.metalness, d.metalnessMap && (m.metalnessMap.value = d.metalnessMap, t(d.metalnessMap, m.metalnessMapTransform)), m.roughness.value = d.roughness, d.roughnessMap && (m.roughnessMap.value = d.roughnessMap, t(d.roughnessMap, m.roughnessMapTransform)), d.envMap && (m.envMapIntensity.value = d.envMapIntensity);
  }
  function p(m, d, b) {
    m.ior.value = d.ior, d.sheen > 0 && (m.sheenColor.value.copy(d.sheenColor).multiplyScalar(d.sheen), m.sheenRoughness.value = d.sheenRoughness, d.sheenColorMap && (m.sheenColorMap.value = d.sheenColorMap, t(d.sheenColorMap, m.sheenColorMapTransform)), d.sheenRoughnessMap && (m.sheenRoughnessMap.value = d.sheenRoughnessMap, t(d.sheenRoughnessMap, m.sheenRoughnessMapTransform))), d.clearcoat > 0 && (m.clearcoat.value = d.clearcoat, m.clearcoatRoughness.value = d.clearcoatRoughness, d.clearcoatMap && (m.clearcoatMap.value = d.clearcoatMap, t(d.clearcoatMap, m.clearcoatMapTransform)), d.clearcoatRoughnessMap && (m.clearcoatRoughnessMap.value = d.clearcoatRoughnessMap, t(d.clearcoatRoughnessMap, m.clearcoatRoughnessMapTransform)), d.clearcoatNormalMap && (m.clearcoatNormalMap.value = d.clearcoatNormalMap, t(d.clearcoatNormalMap, m.clearcoatNormalMapTransform), m.clearcoatNormalScale.value.copy(d.clearcoatNormalScale), d.side === Wt && m.clearcoatNormalScale.value.negate())), d.dispersion > 0 && (m.dispersion.value = d.dispersion), d.iridescence > 0 && (m.iridescence.value = d.iridescence, m.iridescenceIOR.value = d.iridescenceIOR, m.iridescenceThicknessMinimum.value = d.iridescenceThicknessRange[0], m.iridescenceThicknessMaximum.value = d.iridescenceThicknessRange[1], d.iridescenceMap && (m.iridescenceMap.value = d.iridescenceMap, t(d.iridescenceMap, m.iridescenceMapTransform)), d.iridescenceThicknessMap && (m.iridescenceThicknessMap.value = d.iridescenceThicknessMap, t(d.iridescenceThicknessMap, m.iridescenceThicknessMapTransform))), d.transmission > 0 && (m.transmission.value = d.transmission, m.transmissionSamplerMap.value = b.texture, m.transmissionSamplerSize.value.set(b.width, b.height), d.transmissionMap && (m.transmissionMap.value = d.transmissionMap, t(d.transmissionMap, m.transmissionMapTransform)), m.thickness.value = d.thickness, d.thicknessMap && (m.thicknessMap.value = d.thicknessMap, t(d.thicknessMap, m.thicknessMapTransform)), m.attenuationDistance.value = d.attenuationDistance, m.attenuationColor.value.copy(d.attenuationColor)), d.anisotropy > 0 && (m.anisotropyVector.value.set(d.anisotropy * Math.cos(d.anisotropyRotation), d.anisotropy * Math.sin(d.anisotropyRotation)), d.anisotropyMap && (m.anisotropyMap.value = d.anisotropyMap, t(d.anisotropyMap, m.anisotropyMapTransform))), m.specularIntensity.value = d.specularIntensity, m.specularColor.value.copy(d.specularColor), d.specularColorMap && (m.specularColorMap.value = d.specularColorMap, t(d.specularColorMap, m.specularColorMapTransform)), d.specularIntensityMap && (m.specularIntensityMap.value = d.specularIntensityMap, t(d.specularIntensityMap, m.specularIntensityMapTransform));
  }
  function v(m, d) {
    d.matcap && (m.matcap.value = d.matcap);
  }
  function x(m, d) {
    const b = e.get(d).light;
    m.referencePosition.value.setFromMatrixPosition(b.matrixWorld), m.nearDistance.value = b.shadow.camera.near, m.farDistance.value = b.shadow.camera.far;
  }
  return {
    refreshFogUniforms: i,
    refreshMaterialUniforms: s
  };
}
function QS(n, e, t, i) {
  let s = {}, r = {}, o = [];
  const a = n.getParameter(n.MAX_UNIFORM_BUFFER_BINDINGS);
  function l(b, A) {
    const M = A.program;
    i.uniformBlockBinding(b, M);
  }
  function c(b, A) {
    let M = s[b.id];
    M === void 0 && (v(b), M = u(b), s[b.id] = M, b.addEventListener("dispose", m));
    const R = A.program;
    i.updateUBOMapping(b, R);
    const w = e.render.frame;
    r[b.id] !== w && (f(b), r[b.id] = w);
  }
  function u(b) {
    const A = h();
    b.__bindingPointIndex = A;
    const M = n.createBuffer(), R = b.__size, w = b.usage;
    return n.bindBuffer(n.UNIFORM_BUFFER, M), n.bufferData(n.UNIFORM_BUFFER, R, w), n.bindBuffer(n.UNIFORM_BUFFER, null), n.bindBufferBase(n.UNIFORM_BUFFER, A, M), M;
  }
  function h() {
    for (let b = 0; b < a; b++)
      if (o.indexOf(b) === -1)
        return o.push(b), b;
    return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."), 0;
  }
  function f(b) {
    const A = s[b.id], M = b.uniforms, R = b.__cache;
    n.bindBuffer(n.UNIFORM_BUFFER, A);
    for (let w = 0, D = M.length; w < D; w++) {
      const U = Array.isArray(M[w]) ? M[w] : [M[w]];
      for (let y = 0, S = U.length; y < S; y++) {
        const P = U[y];
        if (p(P, w, y, R) === !0) {
          const L = P.__offset, V = Array.isArray(P.value) ? P.value : [P.value];
          let Z = 0;
          for (let te = 0; te < V.length; te++) {
            const $ = V[te], ie = x($);
            typeof $ == "number" || typeof $ == "boolean" ? (P.__data[0] = $, n.bufferSubData(n.UNIFORM_BUFFER, L + Z, P.__data)) : $.isMatrix3 ? (P.__data[0] = $.elements[0], P.__data[1] = $.elements[1], P.__data[2] = $.elements[2], P.__data[3] = 0, P.__data[4] = $.elements[3], P.__data[5] = $.elements[4], P.__data[6] = $.elements[5], P.__data[7] = 0, P.__data[8] = $.elements[6], P.__data[9] = $.elements[7], P.__data[10] = $.elements[8], P.__data[11] = 0) : ($.toArray(P.__data, Z), Z += ie.storage / Float32Array.BYTES_PER_ELEMENT);
          }
          n.bufferSubData(n.UNIFORM_BUFFER, L, P.__data);
        }
      }
    }
    n.bindBuffer(n.UNIFORM_BUFFER, null);
  }
  function p(b, A, M, R) {
    const w = b.value, D = A + "_" + M;
    if (R[D] === void 0)
      return typeof w == "number" || typeof w == "boolean" ? R[D] = w : R[D] = w.clone(), !0;
    {
      const U = R[D];
      if (typeof w == "number" || typeof w == "boolean") {
        if (U !== w)
          return R[D] = w, !0;
      } else if (U.equals(w) === !1)
        return U.copy(w), !0;
    }
    return !1;
  }
  function v(b) {
    const A = b.uniforms;
    let M = 0;
    const R = 16;
    for (let D = 0, U = A.length; D < U; D++) {
      const y = Array.isArray(A[D]) ? A[D] : [A[D]];
      for (let S = 0, P = y.length; S < P; S++) {
        const L = y[S], V = Array.isArray(L.value) ? L.value : [L.value];
        for (let Z = 0, te = V.length; Z < te; Z++) {
          const $ = V[Z], ie = x($), H = M % R, fe = H % ie.boundary, xe = H + fe;
          M += fe, xe !== 0 && R - xe < ie.storage && (M += R - xe), L.__data = new Float32Array(ie.storage / Float32Array.BYTES_PER_ELEMENT), L.__offset = M, M += ie.storage;
        }
      }
    }
    const w = M % R;
    return w > 0 && (M += R - w), b.__size = M, b.__cache = {}, this;
  }
  function x(b) {
    const A = {
      boundary: 0,
      // bytes
      storage: 0
      // bytes
    };
    return typeof b == "number" || typeof b == "boolean" ? (A.boundary = 4, A.storage = 4) : b.isVector2 ? (A.boundary = 8, A.storage = 8) : b.isVector3 || b.isColor ? (A.boundary = 16, A.storage = 12) : b.isVector4 ? (A.boundary = 16, A.storage = 16) : b.isMatrix3 ? (A.boundary = 48, A.storage = 48) : b.isMatrix4 ? (A.boundary = 64, A.storage = 64) : b.isTexture ? console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group.") : console.warn("THREE.WebGLRenderer: Unsupported uniform value type.", b), A;
  }
  function m(b) {
    const A = b.target;
    A.removeEventListener("dispose", m);
    const M = o.indexOf(A.__bindingPointIndex);
    o.splice(M, 1), n.deleteBuffer(s[A.id]), delete s[A.id], delete r[A.id];
  }
  function d() {
    for (const b in s)
      n.deleteBuffer(s[b]);
    o = [], s = {}, r = {};
  }
  return {
    bind: l,
    update: c,
    dispose: d
  };
}
class ey {
  /**
   * Constructs a new WebGL renderer.
   *
   * @param {WebGLRenderer~Options} [parameters] - The configuration parameter.
   */
  constructor(e = {}) {
    const {
      canvas: t = Cg(),
      context: i = null,
      depth: s = !0,
      stencil: r = !1,
      alpha: o = !1,
      antialias: a = !1,
      premultipliedAlpha: l = !0,
      preserveDrawingBuffer: c = !1,
      powerPreference: u = "default",
      failIfMajorPerformanceCaveat: h = !1,
      reversedDepthBuffer: f = !1
    } = e;
    this.isWebGLRenderer = !0;
    let p;
    if (i !== null) {
      if (typeof WebGLRenderingContext < "u" && i instanceof WebGLRenderingContext)
        throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");
      p = i.getContextAttributes().alpha;
    } else
      p = o;
    const v = new Uint32Array(4), x = new Int32Array(4);
    let m = null, d = null;
    const b = [], A = [];
    this.domElement = t, this.debug = {
      /**
       * Enables error checking and reporting when shader programs are being compiled.
       * @type {boolean}
       */
      checkShaderErrors: !0,
      /**
       * Callback for custom error reporting.
       * @type {?Function}
       */
      onShaderError: null
    }, this.autoClear = !0, this.autoClearColor = !0, this.autoClearDepth = !0, this.autoClearStencil = !0, this.sortObjects = !0, this.clippingPlanes = [], this.localClippingEnabled = !1, this.toneMapping = xi, this.toneMappingExposure = 1, this.transmissionResolutionScale = 1;
    const M = this;
    let R = !1;
    this._outputColorSpace = tn;
    let w = 0, D = 0, U = null, y = -1, S = null;
    const P = new lt(), L = new lt();
    let V = null;
    const Z = new We(0);
    let te = 0, $ = t.width, ie = t.height, H = 1, fe = null, xe = null;
    const me = new lt(0, 0, $, ie), de = new lt(0, 0, $, ie);
    let Le = !1;
    const tt = new Ac();
    let Ze = !1, ne = !1;
    const re = new pt(), Ae = new N(), Oe = new lt(), Pe = { background: null, fog: null, environment: null, overrideMaterial: null, isScene: !0 };
    let $e = !1;
    function C() {
      return U === null ? H : 1;
    }
    let g = i;
    function W(T, O) {
      return t.getContext(T, O);
    }
    try {
      const T = {
        alpha: !0,
        depth: s,
        stencil: r,
        antialias: a,
        premultipliedAlpha: l,
        preserveDrawingBuffer: c,
        powerPreference: u,
        failIfMajorPerformanceCaveat: h
      };
      if ("setAttribute" in t && t.setAttribute("data-engine", `three.js r${vc}`), t.addEventListener("webglcontextlost", pe, !1), t.addEventListener("webglcontextrestored", Re, !1), t.addEventListener("webglcontextcreationerror", ce, !1), g === null) {
        const O = "webgl2";
        if (g = W(O, T), g === null)
          throw W(O) ? new Error("Error creating WebGL context with your selected attributes.") : new Error("Error creating WebGL context.");
      }
    } catch (T) {
      throw console.error("THREE.WebGLRenderer: " + T.message), T;
    }
    let j, X, z, ae, q, Q, ee, Se, E, _, I, k, J, G, _e, oe, Ee, Te, le, Me, Ce, be, ge, ke;
    function F() {
      j = new uM(g), j.init(), be = new YS(g, j), X = new iM(g, j, e, be), z = new WS(g, j), X.reversedDepthBuffer && f && z.buffers.depth.setReversed(!0), ae = new dM(g), q = new DS(), Q = new XS(g, j, z, q, X, be, ae), ee = new rM(M), Se = new cM(M), E = new x0(g), ge = new tM(g, E), _ = new hM(g, E, ae, ge), I = new mM(g, _, E, ae), le = new pM(g, X, Q), oe = new sM(q), k = new PS(M, ee, Se, j, X, ge, oe), J = new JS(M, q), G = new IS(), _e = new zS(j), Te = new eM(M, ee, Se, z, I, p, l), Ee = new kS(M, I, X), ke = new QS(g, ae, X, z), Me = new nM(g, j, ae), Ce = new fM(g, j, ae), ae.programs = k.programs, M.capabilities = X, M.extensions = j, M.properties = q, M.renderLists = G, M.shadowMap = Ee, M.state = z, M.info = ae;
    }
    F();
    const he = new $S(M, g);
    this.xr = he, this.getContext = function() {
      return g;
    }, this.getContextAttributes = function() {
      return g.getContextAttributes();
    }, this.forceContextLoss = function() {
      const T = j.get("WEBGL_lose_context");
      T && T.loseContext();
    }, this.forceContextRestore = function() {
      const T = j.get("WEBGL_lose_context");
      T && T.restoreContext();
    }, this.getPixelRatio = function() {
      return H;
    }, this.setPixelRatio = function(T) {
      T !== void 0 && (H = T, this.setSize($, ie, !1));
    }, this.getSize = function(T) {
      return T.set($, ie);
    }, this.setSize = function(T, O, Y = !0) {
      if (he.isPresenting) {
        console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");
        return;
      }
      $ = T, ie = O, t.width = Math.floor(T * H), t.height = Math.floor(O * H), Y === !0 && (t.style.width = T + "px", t.style.height = O + "px"), this.setViewport(0, 0, T, O);
    }, this.getDrawingBufferSize = function(T) {
      return T.set($ * H, ie * H).floor();
    }, this.setDrawingBufferSize = function(T, O, Y) {
      $ = T, ie = O, H = Y, t.width = Math.floor(T * Y), t.height = Math.floor(O * Y), this.setViewport(0, 0, T, O);
    }, this.getCurrentViewport = function(T) {
      return T.copy(P);
    }, this.getViewport = function(T) {
      return T.copy(me);
    }, this.setViewport = function(T, O, Y, K) {
      T.isVector4 ? me.set(T.x, T.y, T.z, T.w) : me.set(T, O, Y, K), z.viewport(P.copy(me).multiplyScalar(H).round());
    }, this.getScissor = function(T) {
      return T.copy(de);
    }, this.setScissor = function(T, O, Y, K) {
      T.isVector4 ? de.set(T.x, T.y, T.z, T.w) : de.set(T, O, Y, K), z.scissor(L.copy(de).multiplyScalar(H).round());
    }, this.getScissorTest = function() {
      return Le;
    }, this.setScissorTest = function(T) {
      z.setScissorTest(Le = T);
    }, this.setOpaqueSort = function(T) {
      fe = T;
    }, this.setTransparentSort = function(T) {
      xe = T;
    }, this.getClearColor = function(T) {
      return T.copy(Te.getClearColor());
    }, this.setClearColor = function() {
      Te.setClearColor(...arguments);
    }, this.getClearAlpha = function() {
      return Te.getClearAlpha();
    }, this.setClearAlpha = function() {
      Te.setClearAlpha(...arguments);
    }, this.clear = function(T = !0, O = !0, Y = !0) {
      let K = 0;
      if (T) {
        let B = !1;
        if (U !== null) {
          const ue = U.texture.format;
          B = ue === Tc || ue === Ec || ue === yc;
        }
        if (B) {
          const ue = U.texture.type, ye = ue === Bn || ue === Xi || ue === Tr || ue === br || ue === Mc || ue === Sc, De = Te.getClearColor(), we = Te.getClearAlpha(), Fe = De.r, He = De.g, Ue = De.b;
          ye ? (v[0] = Fe, v[1] = He, v[2] = Ue, v[3] = we, g.clearBufferuiv(g.COLOR, 0, v)) : (x[0] = Fe, x[1] = He, x[2] = Ue, x[3] = we, g.clearBufferiv(g.COLOR, 0, x));
        } else
          K |= g.COLOR_BUFFER_BIT;
      }
      O && (K |= g.DEPTH_BUFFER_BIT), Y && (K |= g.STENCIL_BUFFER_BIT, this.state.buffers.stencil.setMask(4294967295)), g.clear(K);
    }, this.clearColor = function() {
      this.clear(!0, !1, !1);
    }, this.clearDepth = function() {
      this.clear(!1, !0, !1);
    }, this.clearStencil = function() {
      this.clear(!1, !1, !0);
    }, this.dispose = function() {
      t.removeEventListener("webglcontextlost", pe, !1), t.removeEventListener("webglcontextrestored", Re, !1), t.removeEventListener("webglcontextcreationerror", ce, !1), Te.dispose(), G.dispose(), _e.dispose(), q.dispose(), ee.dispose(), Se.dispose(), I.dispose(), ge.dispose(), ke.dispose(), k.dispose(), he.dispose(), he.removeEventListener("sessionstart", bn), he.removeEventListener("sessionend", Bc), Ei.stop();
    };
    function pe(T) {
      T.preventDefault(), console.log("THREE.WebGLRenderer: Context Lost."), R = !0;
    }
    function Re() {
      console.log("THREE.WebGLRenderer: Context Restored."), R = !1;
      const T = ae.autoReset, O = Ee.enabled, Y = Ee.autoUpdate, K = Ee.needsUpdate, B = Ee.type;
      F(), ae.autoReset = T, Ee.enabled = O, Ee.autoUpdate = Y, Ee.needsUpdate = K, Ee.type = B;
    }
    function ce(T) {
      console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ", T.statusMessage);
    }
    function se(T) {
      const O = T.target;
      O.removeEventListener("dispose", se), Ie(O);
    }
    function Ie(T) {
      Ge(T), q.remove(T);
    }
    function Ge(T) {
      const O = q.get(T).programs;
      O !== void 0 && (O.forEach(function(Y) {
        k.releaseProgram(Y);
      }), T.isShaderMaterial && k.releaseShaderCache(T));
    }
    this.renderBufferDirect = function(T, O, Y, K, B, ue) {
      O === null && (O = Pe);
      const ye = B.isMesh && B.matrixWorld.determinant() < 0, De = Hd(T, O, Y, K, B);
      z.setMaterial(K, ye);
      let we = Y.index, Fe = 1;
      if (K.wireframe === !0) {
        if (we = _.getWireframeAttribute(Y), we === void 0) return;
        Fe = 2;
      }
      const He = Y.drawRange, Ue = Y.attributes.position;
      let Ke = He.start * Fe, rt = (He.start + He.count) * Fe;
      ue !== null && (Ke = Math.max(Ke, ue.start * Fe), rt = Math.min(rt, (ue.start + ue.count) * Fe)), we !== null ? (Ke = Math.max(Ke, 0), rt = Math.min(rt, we.count)) : Ue != null && (Ke = Math.max(Ke, 0), rt = Math.min(rt, Ue.count));
      const Mt = rt - Ke;
      if (Mt < 0 || Mt === 1 / 0) return;
      ge.setup(B, K, De, Y, we);
      let dt, ct = Me;
      if (we !== null && (dt = E.get(we), ct = Ce, ct.setIndex(dt)), B.isMesh)
        K.wireframe === !0 ? (z.setLineWidth(K.wireframeLinewidth * C()), ct.setMode(g.LINES)) : ct.setMode(g.TRIANGLES);
      else if (B.isLine) {
        let Ne = K.linewidth;
        Ne === void 0 && (Ne = 1), z.setLineWidth(Ne * C()), B.isLineSegments ? ct.setMode(g.LINES) : B.isLineLoop ? ct.setMode(g.LINE_LOOP) : ct.setMode(g.LINE_STRIP);
      } else B.isPoints ? ct.setMode(g.POINTS) : B.isSprite && ct.setMode(g.TRIANGLES);
      if (B.isBatchedMesh)
        if (B._multiDrawInstances !== null)
          Rr("THREE.WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."), ct.renderMultiDrawInstances(B._multiDrawStarts, B._multiDrawCounts, B._multiDrawCount, B._multiDrawInstances);
        else if (j.get("WEBGL_multi_draw"))
          ct.renderMultiDraw(B._multiDrawStarts, B._multiDrawCounts, B._multiDrawCount);
        else {
          const Ne = B._multiDrawStarts, _t = B._multiDrawCounts, Je = B._multiDrawCount, Zt = we ? E.get(we).bytesPerElement : 1, Qi = q.get(K).currentProgram.getUniforms();
          for (let Jt = 0; Jt < Je; Jt++)
            Qi.setValue(g, "_gl_DrawID", Jt), ct.render(Ne[Jt] / Zt, _t[Jt]);
        }
      else if (B.isInstancedMesh)
        ct.renderInstances(Ke, Mt, B.count);
      else if (Y.isInstancedBufferGeometry) {
        const Ne = Y._maxInstanceCount !== void 0 ? Y._maxInstanceCount : 1 / 0, _t = Math.min(Y.instanceCount, Ne);
        ct.renderInstances(Ke, Mt, _t);
      } else
        ct.render(Ke, Mt);
    };
    function ht(T, O, Y) {
      T.transparent === !0 && T.side === Qn && T.forceSinglePass === !1 ? (T.side = Wt, T.needsUpdate = !0, Or(T, O, Y), T.side = Si, T.needsUpdate = !0, Or(T, O, Y), T.side = Qn) : Or(T, O, Y);
    }
    this.compile = function(T, O, Y = null) {
      Y === null && (Y = T), d = _e.get(Y), d.init(O), A.push(d), Y.traverseVisible(function(B) {
        B.isLight && B.layers.test(O.layers) && (d.pushLight(B), B.castShadow && d.pushShadow(B));
      }), T !== Y && T.traverseVisible(function(B) {
        B.isLight && B.layers.test(O.layers) && (d.pushLight(B), B.castShadow && d.pushShadow(B));
      }), d.setupLights();
      const K = /* @__PURE__ */ new Set();
      return T.traverse(function(B) {
        if (!(B.isMesh || B.isPoints || B.isLine || B.isSprite))
          return;
        const ue = B.material;
        if (ue)
          if (Array.isArray(ue))
            for (let ye = 0; ye < ue.length; ye++) {
              const De = ue[ye];
              ht(De, Y, B), K.add(De);
            }
          else
            ht(ue, Y, B), K.add(ue);
      }), d = A.pop(), K;
    }, this.compileAsync = function(T, O, Y = null) {
      const K = this.compile(T, O, Y);
      return new Promise((B) => {
        function ue() {
          if (K.forEach(function(ye) {
            q.get(ye).currentProgram.isReady() && K.delete(ye);
          }), K.size === 0) {
            B(T);
            return;
          }
          setTimeout(ue, 10);
        }
        j.get("KHR_parallel_shader_compile") !== null ? ue() : setTimeout(ue, 10);
      });
    };
    let nt = null;
    function Hn(T) {
      nt && nt(T);
    }
    function bn() {
      Ei.stop();
    }
    function Bc() {
      Ei.start();
    }
    const Ei = new wd();
    Ei.setAnimationLoop(Hn), typeof self < "u" && Ei.setContext(self), this.setAnimationLoop = function(T) {
      nt = T, he.setAnimationLoop(T), T === null ? Ei.stop() : Ei.start();
    }, he.addEventListener("sessionstart", bn), he.addEventListener("sessionend", Bc), this.render = function(T, O) {
      if (O !== void 0 && O.isCamera !== !0) {
        console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");
        return;
      }
      if (R === !0) return;
      if (T.matrixWorldAutoUpdate === !0 && T.updateMatrixWorld(), O.parent === null && O.matrixWorldAutoUpdate === !0 && O.updateMatrixWorld(), he.enabled === !0 && he.isPresenting === !0 && (he.cameraAutoUpdate === !0 && he.updateCamera(O), O = he.getCamera()), T.isScene === !0 && T.onBeforeRender(M, T, O, U), d = _e.get(T, A.length), d.init(O), A.push(d), re.multiplyMatrices(O.projectionMatrix, O.matrixWorldInverse), tt.setFromProjectionMatrix(re, Nn, O.reversedDepth), ne = this.localClippingEnabled, Ze = oe.init(this.clippingPlanes, ne), m = G.get(T, b.length), m.init(), b.push(m), he.enabled === !0 && he.isPresenting === !0) {
        const ue = M.xr.getDepthSensingMesh();
        ue !== null && sa(ue, O, -1 / 0, M.sortObjects);
      }
      sa(T, O, 0, M.sortObjects), m.finish(), M.sortObjects === !0 && m.sort(fe, xe), $e = he.enabled === !1 || he.isPresenting === !1 || he.hasDepthSensing() === !1, $e && Te.addToRenderList(m, T), this.info.render.frame++, Ze === !0 && oe.beginShadows();
      const Y = d.state.shadowsArray;
      Ee.render(Y, T, O), Ze === !0 && oe.endShadows(), this.info.autoReset === !0 && this.info.reset();
      const K = m.opaque, B = m.transmissive;
      if (d.setupLights(), O.isArrayCamera) {
        const ue = O.cameras;
        if (B.length > 0)
          for (let ye = 0, De = ue.length; ye < De; ye++) {
            const we = ue[ye];
            Hc(K, B, T, we);
          }
        $e && Te.render(T);
        for (let ye = 0, De = ue.length; ye < De; ye++) {
          const we = ue[ye];
          zc(m, T, we, we.viewport);
        }
      } else
        B.length > 0 && Hc(K, B, T, O), $e && Te.render(T), zc(m, T, O);
      U !== null && D === 0 && (Q.updateMultisampleRenderTarget(U), Q.updateRenderTargetMipmap(U)), T.isScene === !0 && T.onAfterRender(M, T, O), ge.resetDefaultState(), y = -1, S = null, A.pop(), A.length > 0 ? (d = A[A.length - 1], Ze === !0 && oe.setGlobalState(M.clippingPlanes, d.state.camera)) : d = null, b.pop(), b.length > 0 ? m = b[b.length - 1] : m = null;
    };
    function sa(T, O, Y, K) {
      if (T.visible === !1) return;
      if (T.layers.test(O.layers)) {
        if (T.isGroup)
          Y = T.renderOrder;
        else if (T.isLOD)
          T.autoUpdate === !0 && T.update(O);
        else if (T.isLight)
          d.pushLight(T), T.castShadow && d.pushShadow(T);
        else if (T.isSprite) {
          if (!T.frustumCulled || tt.intersectsSprite(T)) {
            K && Oe.setFromMatrixPosition(T.matrixWorld).applyMatrix4(re);
            const ye = I.update(T), De = T.material;
            De.visible && m.push(T, ye, De, Y, Oe.z, null);
          }
        } else if ((T.isMesh || T.isLine || T.isPoints) && (!T.frustumCulled || tt.intersectsObject(T))) {
          const ye = I.update(T), De = T.material;
          if (K && (T.boundingSphere !== void 0 ? (T.boundingSphere === null && T.computeBoundingSphere(), Oe.copy(T.boundingSphere.center)) : (ye.boundingSphere === null && ye.computeBoundingSphere(), Oe.copy(ye.boundingSphere.center)), Oe.applyMatrix4(T.matrixWorld).applyMatrix4(re)), Array.isArray(De)) {
            const we = ye.groups;
            for (let Fe = 0, He = we.length; Fe < He; Fe++) {
              const Ue = we[Fe], Ke = De[Ue.materialIndex];
              Ke && Ke.visible && m.push(T, ye, Ke, Y, Oe.z, Ue);
            }
          } else De.visible && m.push(T, ye, De, Y, Oe.z, null);
        }
      }
      const ue = T.children;
      for (let ye = 0, De = ue.length; ye < De; ye++)
        sa(ue[ye], O, Y, K);
    }
    function zc(T, O, Y, K) {
      const B = T.opaque, ue = T.transmissive, ye = T.transparent;
      d.setupLightsView(Y), Ze === !0 && oe.setGlobalState(M.clippingPlanes, Y), K && z.viewport(P.copy(K)), B.length > 0 && Fr(B, O, Y), ue.length > 0 && Fr(ue, O, Y), ye.length > 0 && Fr(ye, O, Y), z.buffers.depth.setTest(!0), z.buffers.depth.setMask(!0), z.buffers.color.setMask(!0), z.setPolygonOffset(!1);
    }
    function Hc(T, O, Y, K) {
      if ((Y.isScene === !0 ? Y.overrideMaterial : null) !== null)
        return;
      d.state.transmissionRenderTarget[K.id] === void 0 && (d.state.transmissionRenderTarget[K.id] = new qi(1, 1, {
        generateMipmaps: !0,
        type: j.has("EXT_color_buffer_half_float") || j.has("EXT_color_buffer_float") ? Lr : Bn,
        minFilter: Vi,
        samples: 4,
        stencilBuffer: r,
        resolveDepthBuffer: !1,
        resolveStencilBuffer: !1,
        colorSpace: Qe.workingColorSpace
      }));
      const ue = d.state.transmissionRenderTarget[K.id], ye = K.viewport || P;
      ue.setSize(ye.z * M.transmissionResolutionScale, ye.w * M.transmissionResolutionScale);
      const De = M.getRenderTarget(), we = M.getActiveCubeFace(), Fe = M.getActiveMipmapLevel();
      M.setRenderTarget(ue), M.getClearColor(Z), te = M.getClearAlpha(), te < 1 && M.setClearColor(16777215, 0.5), M.clear(), $e && Te.render(Y);
      const He = M.toneMapping;
      M.toneMapping = xi;
      const Ue = K.viewport;
      if (K.viewport !== void 0 && (K.viewport = void 0), d.setupLightsView(K), Ze === !0 && oe.setGlobalState(M.clippingPlanes, K), Fr(T, Y, K), Q.updateMultisampleRenderTarget(ue), Q.updateRenderTargetMipmap(ue), j.has("WEBGL_multisampled_render_to_texture") === !1) {
        let Ke = !1;
        for (let rt = 0, Mt = O.length; rt < Mt; rt++) {
          const dt = O[rt], ct = dt.object, Ne = dt.geometry, _t = dt.material, Je = dt.group;
          if (_t.side === Qn && ct.layers.test(K.layers)) {
            const Zt = _t.side;
            _t.side = Wt, _t.needsUpdate = !0, Vc(ct, Y, K, Ne, _t, Je), _t.side = Zt, _t.needsUpdate = !0, Ke = !0;
          }
        }
        Ke === !0 && (Q.updateMultisampleRenderTarget(ue), Q.updateRenderTargetMipmap(ue));
      }
      M.setRenderTarget(De, we, Fe), M.setClearColor(Z, te), Ue !== void 0 && (K.viewport = Ue), M.toneMapping = He;
    }
    function Fr(T, O, Y) {
      const K = O.isScene === !0 ? O.overrideMaterial : null;
      for (let B = 0, ue = T.length; B < ue; B++) {
        const ye = T[B], De = ye.object, we = ye.geometry, Fe = ye.group;
        let He = ye.material;
        He.allowOverride === !0 && K !== null && (He = K), De.layers.test(Y.layers) && Vc(De, O, Y, we, He, Fe);
      }
    }
    function Vc(T, O, Y, K, B, ue) {
      T.onBeforeRender(M, O, Y, K, B, ue), T.modelViewMatrix.multiplyMatrices(Y.matrixWorldInverse, T.matrixWorld), T.normalMatrix.getNormalMatrix(T.modelViewMatrix), B.onBeforeRender(M, O, Y, K, T, ue), B.transparent === !0 && B.side === Qn && B.forceSinglePass === !1 ? (B.side = Wt, B.needsUpdate = !0, M.renderBufferDirect(Y, O, K, B, T, ue), B.side = Si, B.needsUpdate = !0, M.renderBufferDirect(Y, O, K, B, T, ue), B.side = Qn) : M.renderBufferDirect(Y, O, K, B, T, ue), T.onAfterRender(M, O, Y, K, B, ue);
    }
    function Or(T, O, Y) {
      O.isScene !== !0 && (O = Pe);
      const K = q.get(T), B = d.state.lights, ue = d.state.shadowsArray, ye = B.state.version, De = k.getParameters(T, B.state, ue, O, Y), we = k.getProgramCacheKey(De);
      let Fe = K.programs;
      K.environment = T.isMeshStandardMaterial ? O.environment : null, K.fog = O.fog, K.envMap = (T.isMeshStandardMaterial ? Se : ee).get(T.envMap || K.environment), K.envMapRotation = K.environment !== null && T.envMap === null ? O.environmentRotation : T.envMapRotation, Fe === void 0 && (T.addEventListener("dispose", se), Fe = /* @__PURE__ */ new Map(), K.programs = Fe);
      let He = Fe.get(we);
      if (He !== void 0) {
        if (K.currentProgram === He && K.lightsStateVersion === ye)
          return Gc(T, De), He;
      } else
        De.uniforms = k.getUniforms(T), T.onBeforeCompile(De, M), He = k.acquireProgram(De, we), Fe.set(we, He), K.uniforms = De.uniforms;
      const Ue = K.uniforms;
      return (!T.isShaderMaterial && !T.isRawShaderMaterial || T.clipping === !0) && (Ue.clippingPlanes = oe.uniform), Gc(T, De), K.needsLights = kd(T), K.lightsStateVersion = ye, K.needsLights && (Ue.ambientLightColor.value = B.state.ambient, Ue.lightProbe.value = B.state.probe, Ue.directionalLights.value = B.state.directional, Ue.directionalLightShadows.value = B.state.directionalShadow, Ue.spotLights.value = B.state.spot, Ue.spotLightShadows.value = B.state.spotShadow, Ue.rectAreaLights.value = B.state.rectArea, Ue.ltc_1.value = B.state.rectAreaLTC1, Ue.ltc_2.value = B.state.rectAreaLTC2, Ue.pointLights.value = B.state.point, Ue.pointLightShadows.value = B.state.pointShadow, Ue.hemisphereLights.value = B.state.hemi, Ue.directionalShadowMap.value = B.state.directionalShadowMap, Ue.directionalShadowMatrix.value = B.state.directionalShadowMatrix, Ue.spotShadowMap.value = B.state.spotShadowMap, Ue.spotLightMatrix.value = B.state.spotLightMatrix, Ue.spotLightMap.value = B.state.spotLightMap, Ue.pointShadowMap.value = B.state.pointShadowMap, Ue.pointShadowMatrix.value = B.state.pointShadowMatrix), K.currentProgram = He, K.uniformsList = null, He;
    }
    function kc(T) {
      if (T.uniformsList === null) {
        const O = T.currentProgram.getUniforms();
        T.uniformsList = Ao.seqWithValue(O.seq, T.uniforms);
      }
      return T.uniformsList;
    }
    function Gc(T, O) {
      const Y = q.get(T);
      Y.outputColorSpace = O.outputColorSpace, Y.batching = O.batching, Y.batchingColor = O.batchingColor, Y.instancing = O.instancing, Y.instancingColor = O.instancingColor, Y.instancingMorph = O.instancingMorph, Y.skinning = O.skinning, Y.morphTargets = O.morphTargets, Y.morphNormals = O.morphNormals, Y.morphColors = O.morphColors, Y.morphTargetsCount = O.morphTargetsCount, Y.numClippingPlanes = O.numClippingPlanes, Y.numIntersection = O.numClipIntersection, Y.vertexAlphas = O.vertexAlphas, Y.vertexTangents = O.vertexTangents, Y.toneMapping = O.toneMapping;
    }
    function Hd(T, O, Y, K, B) {
      O.isScene !== !0 && (O = Pe), Q.resetTextureUnits();
      const ue = O.fog, ye = K.isMeshStandardMaterial ? O.environment : null, De = U === null ? M.outputColorSpace : U.isXRRenderTarget === !0 ? U.texture.colorSpace : Os, we = (K.isMeshStandardMaterial ? Se : ee).get(K.envMap || ye), Fe = K.vertexColors === !0 && !!Y.attributes.color && Y.attributes.color.itemSize === 4, He = !!Y.attributes.tangent && (!!K.normalMap || K.anisotropy > 0), Ue = !!Y.morphAttributes.position, Ke = !!Y.morphAttributes.normal, rt = !!Y.morphAttributes.color;
      let Mt = xi;
      K.toneMapped && (U === null || U.isXRRenderTarget === !0) && (Mt = M.toneMapping);
      const dt = Y.morphAttributes.position || Y.morphAttributes.normal || Y.morphAttributes.color, ct = dt !== void 0 ? dt.length : 0, Ne = q.get(K), _t = d.state.lights;
      if (Ze === !0 && (ne === !0 || T !== S)) {
        const Ft = T === S && K.id === y;
        oe.setState(K, T, Ft);
      }
      let Je = !1;
      K.version === Ne.__version ? (Ne.needsLights && Ne.lightsStateVersion !== _t.state.version || Ne.outputColorSpace !== De || B.isBatchedMesh && Ne.batching === !1 || !B.isBatchedMesh && Ne.batching === !0 || B.isBatchedMesh && Ne.batchingColor === !0 && B.colorTexture === null || B.isBatchedMesh && Ne.batchingColor === !1 && B.colorTexture !== null || B.isInstancedMesh && Ne.instancing === !1 || !B.isInstancedMesh && Ne.instancing === !0 || B.isSkinnedMesh && Ne.skinning === !1 || !B.isSkinnedMesh && Ne.skinning === !0 || B.isInstancedMesh && Ne.instancingColor === !0 && B.instanceColor === null || B.isInstancedMesh && Ne.instancingColor === !1 && B.instanceColor !== null || B.isInstancedMesh && Ne.instancingMorph === !0 && B.morphTexture === null || B.isInstancedMesh && Ne.instancingMorph === !1 && B.morphTexture !== null || Ne.envMap !== we || K.fog === !0 && Ne.fog !== ue || Ne.numClippingPlanes !== void 0 && (Ne.numClippingPlanes !== oe.numPlanes || Ne.numIntersection !== oe.numIntersection) || Ne.vertexAlphas !== Fe || Ne.vertexTangents !== He || Ne.morphTargets !== Ue || Ne.morphNormals !== Ke || Ne.morphColors !== rt || Ne.toneMapping !== Mt || Ne.morphTargetsCount !== ct) && (Je = !0) : (Je = !0, Ne.__version = K.version);
      let Zt = Ne.currentProgram;
      Je === !0 && (Zt = Or(K, O, B));
      let Qi = !1, Jt = !1, ks = !1;
      const gt = Zt.getUniforms(), rn = Ne.uniforms;
      if (z.useProgram(Zt.program) && (Qi = !0, Jt = !0, ks = !0), K.id !== y && (y = K.id, Jt = !0), Qi || S !== T) {
        z.buffers.depth.getReversed() && T.reversedDepth !== !0 && (T._reversedDepth = !0, T.updateProjectionMatrix()), gt.setValue(g, "projectionMatrix", T.projectionMatrix), gt.setValue(g, "viewMatrix", T.matrixWorldInverse);
        const Xt = gt.map.cameraPosition;
        Xt !== void 0 && Xt.setValue(g, Ae.setFromMatrixPosition(T.matrixWorld)), X.logarithmicDepthBuffer && gt.setValue(
          g,
          "logDepthBufFC",
          2 / (Math.log(T.far + 1) / Math.LN2)
        ), (K.isMeshPhongMaterial || K.isMeshToonMaterial || K.isMeshLambertMaterial || K.isMeshBasicMaterial || K.isMeshStandardMaterial || K.isShaderMaterial) && gt.setValue(g, "isOrthographic", T.isOrthographicCamera === !0), S !== T && (S = T, Jt = !0, ks = !0);
      }
      if (B.isSkinnedMesh) {
        gt.setOptional(g, B, "bindMatrix"), gt.setOptional(g, B, "bindMatrixInverse");
        const Ft = B.skeleton;
        Ft && (Ft.boneTexture === null && Ft.computeBoneTexture(), gt.setValue(g, "boneTexture", Ft.boneTexture, Q));
      }
      B.isBatchedMesh && (gt.setOptional(g, B, "batchingTexture"), gt.setValue(g, "batchingTexture", B._matricesTexture, Q), gt.setOptional(g, B, "batchingIdTexture"), gt.setValue(g, "batchingIdTexture", B._indirectTexture, Q), gt.setOptional(g, B, "batchingColorTexture"), B._colorsTexture !== null && gt.setValue(g, "batchingColorTexture", B._colorsTexture, Q));
      const on = Y.morphAttributes;
      if ((on.position !== void 0 || on.normal !== void 0 || on.color !== void 0) && le.update(B, Y, Zt), (Jt || Ne.receiveShadow !== B.receiveShadow) && (Ne.receiveShadow = B.receiveShadow, gt.setValue(g, "receiveShadow", B.receiveShadow)), K.isMeshGouraudMaterial && K.envMap !== null && (rn.envMap.value = we, rn.flipEnvMap.value = we.isCubeTexture && we.isRenderTargetTexture === !1 ? -1 : 1), K.isMeshStandardMaterial && K.envMap === null && O.environment !== null && (rn.envMapIntensity.value = O.environmentIntensity), Jt && (gt.setValue(g, "toneMappingExposure", M.toneMappingExposure), Ne.needsLights && Vd(rn, ks), ue && K.fog === !0 && J.refreshFogUniforms(rn, ue), J.refreshMaterialUniforms(rn, K, H, ie, d.state.transmissionRenderTarget[T.id]), Ao.upload(g, kc(Ne), rn, Q)), K.isShaderMaterial && K.uniformsNeedUpdate === !0 && (Ao.upload(g, kc(Ne), rn, Q), K.uniformsNeedUpdate = !1), K.isSpriteMaterial && gt.setValue(g, "center", B.center), gt.setValue(g, "modelViewMatrix", B.modelViewMatrix), gt.setValue(g, "normalMatrix", B.normalMatrix), gt.setValue(g, "modelMatrix", B.matrixWorld), K.isShaderMaterial || K.isRawShaderMaterial) {
        const Ft = K.uniformsGroups;
        for (let Xt = 0, ra = Ft.length; Xt < ra; Xt++) {
          const Ti = Ft[Xt];
          ke.update(Ti, Zt), ke.bind(Ti, Zt);
        }
      }
      return Zt;
    }
    function Vd(T, O) {
      T.ambientLightColor.needsUpdate = O, T.lightProbe.needsUpdate = O, T.directionalLights.needsUpdate = O, T.directionalLightShadows.needsUpdate = O, T.pointLights.needsUpdate = O, T.pointLightShadows.needsUpdate = O, T.spotLights.needsUpdate = O, T.spotLightShadows.needsUpdate = O, T.rectAreaLights.needsUpdate = O, T.hemisphereLights.needsUpdate = O;
    }
    function kd(T) {
      return T.isMeshLambertMaterial || T.isMeshToonMaterial || T.isMeshPhongMaterial || T.isMeshStandardMaterial || T.isShadowMaterial || T.isShaderMaterial && T.lights === !0;
    }
    this.getActiveCubeFace = function() {
      return w;
    }, this.getActiveMipmapLevel = function() {
      return D;
    }, this.getRenderTarget = function() {
      return U;
    }, this.setRenderTargetTextures = function(T, O, Y) {
      const K = q.get(T);
      K.__autoAllocateDepthBuffer = T.resolveDepthBuffer === !1, K.__autoAllocateDepthBuffer === !1 && (K.__useRenderToTexture = !1), q.get(T.texture).__webglTexture = O, q.get(T.depthTexture).__webglTexture = K.__autoAllocateDepthBuffer ? void 0 : Y, K.__hasExternalTextures = !0;
    }, this.setRenderTargetFramebuffer = function(T, O) {
      const Y = q.get(T);
      Y.__webglFramebuffer = O, Y.__useDefaultFramebuffer = O === void 0;
    };
    const Gd = g.createFramebuffer();
    this.setRenderTarget = function(T, O = 0, Y = 0) {
      U = T, w = O, D = Y;
      let K = !0, B = null, ue = !1, ye = !1;
      if (T) {
        const we = q.get(T);
        if (we.__useDefaultFramebuffer !== void 0)
          z.bindFramebuffer(g.FRAMEBUFFER, null), K = !1;
        else if (we.__webglFramebuffer === void 0)
          Q.setupRenderTarget(T);
        else if (we.__hasExternalTextures)
          Q.rebindTextures(T, q.get(T.texture).__webglTexture, q.get(T.depthTexture).__webglTexture);
        else if (T.depthBuffer) {
          const Ue = T.depthTexture;
          if (we.__boundDepthTexture !== Ue) {
            if (Ue !== null && q.has(Ue) && (T.width !== Ue.image.width || T.height !== Ue.image.height))
              throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");
            Q.setupDepthRenderbuffer(T);
          }
        }
        const Fe = T.texture;
        (Fe.isData3DTexture || Fe.isDataArrayTexture || Fe.isCompressedArrayTexture) && (ye = !0);
        const He = q.get(T).__webglFramebuffer;
        T.isWebGLCubeRenderTarget ? (Array.isArray(He[O]) ? B = He[O][Y] : B = He[O], ue = !0) : T.samples > 0 && Q.useMultisampledRTT(T) === !1 ? B = q.get(T).__webglMultisampledFramebuffer : Array.isArray(He) ? B = He[Y] : B = He, P.copy(T.viewport), L.copy(T.scissor), V = T.scissorTest;
      } else
        P.copy(me).multiplyScalar(H).floor(), L.copy(de).multiplyScalar(H).floor(), V = Le;
      if (Y !== 0 && (B = Gd), z.bindFramebuffer(g.FRAMEBUFFER, B) && K && z.drawBuffers(T, B), z.viewport(P), z.scissor(L), z.setScissorTest(V), ue) {
        const we = q.get(T.texture);
        g.framebufferTexture2D(g.FRAMEBUFFER, g.COLOR_ATTACHMENT0, g.TEXTURE_CUBE_MAP_POSITIVE_X + O, we.__webglTexture, Y);
      } else if (ye) {
        const we = O;
        for (let Fe = 0; Fe < T.textures.length; Fe++) {
          const He = q.get(T.textures[Fe]);
          g.framebufferTextureLayer(g.FRAMEBUFFER, g.COLOR_ATTACHMENT0 + Fe, He.__webglTexture, Y, we);
        }
      } else if (T !== null && Y !== 0) {
        const we = q.get(T.texture);
        g.framebufferTexture2D(g.FRAMEBUFFER, g.COLOR_ATTACHMENT0, g.TEXTURE_2D, we.__webglTexture, Y);
      }
      y = -1;
    }, this.readRenderTargetPixels = function(T, O, Y, K, B, ue, ye, De = 0) {
      if (!(T && T.isWebGLRenderTarget)) {
        console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
        return;
      }
      let we = q.get(T).__webglFramebuffer;
      if (T.isWebGLCubeRenderTarget && ye !== void 0 && (we = we[ye]), we) {
        z.bindFramebuffer(g.FRAMEBUFFER, we);
        try {
          const Fe = T.textures[De], He = Fe.format, Ue = Fe.type;
          if (!X.textureFormatReadable(He)) {
            console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");
            return;
          }
          if (!X.textureTypeReadable(Ue)) {
            console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");
            return;
          }
          O >= 0 && O <= T.width - K && Y >= 0 && Y <= T.height - B && (T.textures.length > 1 && g.readBuffer(g.COLOR_ATTACHMENT0 + De), g.readPixels(O, Y, K, B, be.convert(He), be.convert(Ue), ue));
        } finally {
          const Fe = U !== null ? q.get(U).__webglFramebuffer : null;
          z.bindFramebuffer(g.FRAMEBUFFER, Fe);
        }
      }
    }, this.readRenderTargetPixelsAsync = async function(T, O, Y, K, B, ue, ye, De = 0) {
      if (!(T && T.isWebGLRenderTarget))
        throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
      let we = q.get(T).__webglFramebuffer;
      if (T.isWebGLCubeRenderTarget && ye !== void 0 && (we = we[ye]), we)
        if (O >= 0 && O <= T.width - K && Y >= 0 && Y <= T.height - B) {
          z.bindFramebuffer(g.FRAMEBUFFER, we);
          const Fe = T.textures[De], He = Fe.format, Ue = Fe.type;
          if (!X.textureFormatReadable(He))
            throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");
          if (!X.textureTypeReadable(Ue))
            throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");
          const Ke = g.createBuffer();
          g.bindBuffer(g.PIXEL_PACK_BUFFER, Ke), g.bufferData(g.PIXEL_PACK_BUFFER, ue.byteLength, g.STREAM_READ), T.textures.length > 1 && g.readBuffer(g.COLOR_ATTACHMENT0 + De), g.readPixels(O, Y, K, B, be.convert(He), be.convert(Ue), 0);
          const rt = U !== null ? q.get(U).__webglFramebuffer : null;
          z.bindFramebuffer(g.FRAMEBUFFER, rt);
          const Mt = g.fenceSync(g.SYNC_GPU_COMMANDS_COMPLETE, 0);
          return g.flush(), await Pg(g, Mt, 4), g.bindBuffer(g.PIXEL_PACK_BUFFER, Ke), g.getBufferSubData(g.PIXEL_PACK_BUFFER, 0, ue), g.deleteBuffer(Ke), g.deleteSync(Mt), ue;
        } else
          throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.");
    }, this.copyFramebufferToTexture = function(T, O = null, Y = 0) {
      const K = Math.pow(2, -Y), B = Math.floor(T.image.width * K), ue = Math.floor(T.image.height * K), ye = O !== null ? O.x : 0, De = O !== null ? O.y : 0;
      Q.setTexture2D(T, 0), g.copyTexSubImage2D(g.TEXTURE_2D, Y, 0, 0, ye, De, B, ue), z.unbindTexture();
    };
    const Wd = g.createFramebuffer(), Xd = g.createFramebuffer();
    this.copyTextureToTexture = function(T, O, Y = null, K = null, B = 0, ue = null) {
      ue === null && (B !== 0 ? (Rr("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."), ue = B, B = 0) : ue = 0);
      let ye, De, we, Fe, He, Ue, Ke, rt, Mt;
      const dt = T.isCompressedTexture ? T.mipmaps[ue] : T.image;
      if (Y !== null)
        ye = Y.max.x - Y.min.x, De = Y.max.y - Y.min.y, we = Y.isBox3 ? Y.max.z - Y.min.z : 1, Fe = Y.min.x, He = Y.min.y, Ue = Y.isBox3 ? Y.min.z : 0;
      else {
        const on = Math.pow(2, -B);
        ye = Math.floor(dt.width * on), De = Math.floor(dt.height * on), T.isDataArrayTexture ? we = dt.depth : T.isData3DTexture ? we = Math.floor(dt.depth * on) : we = 1, Fe = 0, He = 0, Ue = 0;
      }
      K !== null ? (Ke = K.x, rt = K.y, Mt = K.z) : (Ke = 0, rt = 0, Mt = 0);
      const ct = be.convert(O.format), Ne = be.convert(O.type);
      let _t;
      O.isData3DTexture ? (Q.setTexture3D(O, 0), _t = g.TEXTURE_3D) : O.isDataArrayTexture || O.isCompressedArrayTexture ? (Q.setTexture2DArray(O, 0), _t = g.TEXTURE_2D_ARRAY) : (Q.setTexture2D(O, 0), _t = g.TEXTURE_2D), g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL, O.flipY), g.pixelStorei(g.UNPACK_PREMULTIPLY_ALPHA_WEBGL, O.premultiplyAlpha), g.pixelStorei(g.UNPACK_ALIGNMENT, O.unpackAlignment);
      const Je = g.getParameter(g.UNPACK_ROW_LENGTH), Zt = g.getParameter(g.UNPACK_IMAGE_HEIGHT), Qi = g.getParameter(g.UNPACK_SKIP_PIXELS), Jt = g.getParameter(g.UNPACK_SKIP_ROWS), ks = g.getParameter(g.UNPACK_SKIP_IMAGES);
      g.pixelStorei(g.UNPACK_ROW_LENGTH, dt.width), g.pixelStorei(g.UNPACK_IMAGE_HEIGHT, dt.height), g.pixelStorei(g.UNPACK_SKIP_PIXELS, Fe), g.pixelStorei(g.UNPACK_SKIP_ROWS, He), g.pixelStorei(g.UNPACK_SKIP_IMAGES, Ue);
      const gt = T.isDataArrayTexture || T.isData3DTexture, rn = O.isDataArrayTexture || O.isData3DTexture;
      if (T.isDepthTexture) {
        const on = q.get(T), Ft = q.get(O), Xt = q.get(on.__renderTarget), ra = q.get(Ft.__renderTarget);
        z.bindFramebuffer(g.READ_FRAMEBUFFER, Xt.__webglFramebuffer), z.bindFramebuffer(g.DRAW_FRAMEBUFFER, ra.__webglFramebuffer);
        for (let Ti = 0; Ti < we; Ti++)
          gt && (g.framebufferTextureLayer(g.READ_FRAMEBUFFER, g.COLOR_ATTACHMENT0, q.get(T).__webglTexture, B, Ue + Ti), g.framebufferTextureLayer(g.DRAW_FRAMEBUFFER, g.COLOR_ATTACHMENT0, q.get(O).__webglTexture, ue, Mt + Ti)), g.blitFramebuffer(Fe, He, ye, De, Ke, rt, ye, De, g.DEPTH_BUFFER_BIT, g.NEAREST);
        z.bindFramebuffer(g.READ_FRAMEBUFFER, null), z.bindFramebuffer(g.DRAW_FRAMEBUFFER, null);
      } else if (B !== 0 || T.isRenderTargetTexture || q.has(T)) {
        const on = q.get(T), Ft = q.get(O);
        z.bindFramebuffer(g.READ_FRAMEBUFFER, Wd), z.bindFramebuffer(g.DRAW_FRAMEBUFFER, Xd);
        for (let Xt = 0; Xt < we; Xt++)
          gt ? g.framebufferTextureLayer(g.READ_FRAMEBUFFER, g.COLOR_ATTACHMENT0, on.__webglTexture, B, Ue + Xt) : g.framebufferTexture2D(g.READ_FRAMEBUFFER, g.COLOR_ATTACHMENT0, g.TEXTURE_2D, on.__webglTexture, B), rn ? g.framebufferTextureLayer(g.DRAW_FRAMEBUFFER, g.COLOR_ATTACHMENT0, Ft.__webglTexture, ue, Mt + Xt) : g.framebufferTexture2D(g.DRAW_FRAMEBUFFER, g.COLOR_ATTACHMENT0, g.TEXTURE_2D, Ft.__webglTexture, ue), B !== 0 ? g.blitFramebuffer(Fe, He, ye, De, Ke, rt, ye, De, g.COLOR_BUFFER_BIT, g.NEAREST) : rn ? g.copyTexSubImage3D(_t, ue, Ke, rt, Mt + Xt, Fe, He, ye, De) : g.copyTexSubImage2D(_t, ue, Ke, rt, Fe, He, ye, De);
        z.bindFramebuffer(g.READ_FRAMEBUFFER, null), z.bindFramebuffer(g.DRAW_FRAMEBUFFER, null);
      } else
        rn ? T.isDataTexture || T.isData3DTexture ? g.texSubImage3D(_t, ue, Ke, rt, Mt, ye, De, we, ct, Ne, dt.data) : O.isCompressedArrayTexture ? g.compressedTexSubImage3D(_t, ue, Ke, rt, Mt, ye, De, we, ct, dt.data) : g.texSubImage3D(_t, ue, Ke, rt, Mt, ye, De, we, ct, Ne, dt) : T.isDataTexture ? g.texSubImage2D(g.TEXTURE_2D, ue, Ke, rt, ye, De, ct, Ne, dt.data) : T.isCompressedTexture ? g.compressedTexSubImage2D(g.TEXTURE_2D, ue, Ke, rt, dt.width, dt.height, ct, dt.data) : g.texSubImage2D(g.TEXTURE_2D, ue, Ke, rt, ye, De, ct, Ne, dt);
      g.pixelStorei(g.UNPACK_ROW_LENGTH, Je), g.pixelStorei(g.UNPACK_IMAGE_HEIGHT, Zt), g.pixelStorei(g.UNPACK_SKIP_PIXELS, Qi), g.pixelStorei(g.UNPACK_SKIP_ROWS, Jt), g.pixelStorei(g.UNPACK_SKIP_IMAGES, ks), ue === 0 && O.generateMipmaps && g.generateMipmap(_t), z.unbindTexture();
    }, this.initRenderTarget = function(T) {
      q.get(T).__webglFramebuffer === void 0 && Q.setupRenderTarget(T);
    }, this.initTexture = function(T) {
      T.isCubeTexture ? Q.setTextureCube(T, 0) : T.isData3DTexture ? Q.setTexture3D(T, 0) : T.isDataArrayTexture || T.isCompressedArrayTexture ? Q.setTexture2DArray(T, 0) : Q.setTexture2D(T, 0), z.unbindTexture();
    }, this.resetState = function() {
      w = 0, D = 0, U = null, z.reset(), ge.reset();
    }, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
  }
  /**
   * Defines the coordinate system of the renderer.
   *
   * In `WebGLRenderer`, the value is always `WebGLCoordinateSystem`.
   *
   * @type {WebGLCoordinateSystem|WebGPUCoordinateSystem}
   * @default WebGLCoordinateSystem
   * @readonly
   */
  get coordinateSystem() {
    return Nn;
  }
  /**
   * Defines the output color space of the renderer.
   *
   * @type {SRGBColorSpace|LinearSRGBColorSpace}
   * @default SRGBColorSpace
   */
  get outputColorSpace() {
    return this._outputColorSpace;
  }
  set outputColorSpace(e) {
    this._outputColorSpace = e;
    const t = this.getContext();
    t.drawingBufferColorSpace = Qe._getDrawingBufferColorSpace(e), t.unpackColorSpace = Qe._getUnpackColorSpace();
  }
}
const Dh = { type: "change" }, Ic = { type: "start" }, Ld = { type: "end" }, _o = new na(), Lh = new mi(), ty = Math.cos(70 * Rg.DEG2RAD), Et = new N(), qt = 2 * Math.PI, at = {
  NONE: -1,
  ROTATE: 0,
  DOLLY: 1,
  PAN: 2,
  TOUCH_ROTATE: 3,
  TOUCH_PAN: 4,
  TOUCH_DOLLY_PAN: 5,
  TOUCH_DOLLY_ROTATE: 6
}, Qa = 1e-6;
class ny extends g0 {
  /**
   * Constructs a new controls instance.
   *
   * @param {Object3D} object - The object that is managed by the controls.
   * @param {?HTMLDOMElement} domElement - The HTML element used for event listeners.
   */
  constructor(e, t = null) {
    super(e, t), this.state = at.NONE, this.target = new N(), this.cursor = new N(), this.minDistance = 0, this.maxDistance = 1 / 0, this.minZoom = 0, this.maxZoom = 1 / 0, this.minTargetRadius = 0, this.maxTargetRadius = 1 / 0, this.minPolarAngle = 0, this.maxPolarAngle = Math.PI, this.minAzimuthAngle = -1 / 0, this.maxAzimuthAngle = 1 / 0, this.enableDamping = !1, this.dampingFactor = 0.05, this.enableZoom = !0, this.zoomSpeed = 1, this.enableRotate = !0, this.rotateSpeed = 1, this.keyRotateSpeed = 1, this.enablePan = !0, this.panSpeed = 1, this.screenSpacePanning = !0, this.keyPanSpeed = 7, this.zoomToCursor = !1, this.autoRotate = !1, this.autoRotateSpeed = 2, this.keys = { LEFT: "ArrowLeft", UP: "ArrowUp", RIGHT: "ArrowRight", BOTTOM: "ArrowDown" }, this.mouseButtons = { LEFT: Ps.ROTATE, MIDDLE: Ps.DOLLY, RIGHT: Ps.PAN }, this.touches = { ONE: Ss.ROTATE, TWO: Ss.DOLLY_PAN }, this.target0 = this.target.clone(), this.position0 = this.object.position.clone(), this.zoom0 = this.object.zoom, this._domElementKeyEvents = null, this._lastPosition = new N(), this._lastQuaternion = new Yi(), this._lastTargetPosition = new N(), this._quat = new Yi().setFromUnitVectors(e.up, new N(0, 1, 0)), this._quatInverse = this._quat.clone().invert(), this._spherical = new sh(), this._sphericalDelta = new sh(), this._scale = 1, this._panOffset = new N(), this._rotateStart = new Ve(), this._rotateEnd = new Ve(), this._rotateDelta = new Ve(), this._panStart = new Ve(), this._panEnd = new Ve(), this._panDelta = new Ve(), this._dollyStart = new Ve(), this._dollyEnd = new Ve(), this._dollyDelta = new Ve(), this._dollyDirection = new N(), this._mouse = new Ve(), this._performCursorZoom = !1, this._pointers = [], this._pointerPositions = {}, this._controlActive = !1, this._onPointerMove = sy.bind(this), this._onPointerDown = iy.bind(this), this._onPointerUp = ry.bind(this), this._onContextMenu = fy.bind(this), this._onMouseWheel = ly.bind(this), this._onKeyDown = cy.bind(this), this._onTouchStart = uy.bind(this), this._onTouchMove = hy.bind(this), this._onMouseDown = oy.bind(this), this._onMouseMove = ay.bind(this), this._interceptControlDown = dy.bind(this), this._interceptControlUp = py.bind(this), this.domElement !== null && this.connect(this.domElement), this.update();
  }
  connect(e) {
    super.connect(e), this.domElement.addEventListener("pointerdown", this._onPointerDown), this.domElement.addEventListener("pointercancel", this._onPointerUp), this.domElement.addEventListener("contextmenu", this._onContextMenu), this.domElement.addEventListener("wheel", this._onMouseWheel, { passive: !1 }), this.domElement.getRootNode().addEventListener("keydown", this._interceptControlDown, { passive: !0, capture: !0 }), this.domElement.style.touchAction = "none";
  }
  disconnect() {
    this.domElement.removeEventListener("pointerdown", this._onPointerDown), this.domElement.removeEventListener("pointermove", this._onPointerMove), this.domElement.removeEventListener("pointerup", this._onPointerUp), this.domElement.removeEventListener("pointercancel", this._onPointerUp), this.domElement.removeEventListener("wheel", this._onMouseWheel), this.domElement.removeEventListener("contextmenu", this._onContextMenu), this.stopListenToKeyEvents(), this.domElement.getRootNode().removeEventListener("keydown", this._interceptControlDown, { capture: !0 }), this.domElement.style.touchAction = "auto";
  }
  dispose() {
    this.disconnect();
  }
  /**
   * Get the current vertical rotation, in radians.
   *
   * @return {number} The current vertical rotation, in radians.
   */
  getPolarAngle() {
    return this._spherical.phi;
  }
  /**
   * Get the current horizontal rotation, in radians.
   *
   * @return {number} The current horizontal rotation, in radians.
   */
  getAzimuthalAngle() {
    return this._spherical.theta;
  }
  /**
   * Returns the distance from the camera to the target.
   *
   * @return {number} The distance from the camera to the target.
   */
  getDistance() {
    return this.object.position.distanceTo(this.target);
  }
  /**
   * Adds key event listeners to the given DOM element.
   * `window` is a recommended argument for using this method.
   *
   * @param {HTMLDOMElement} domElement - The DOM element
   */
  listenToKeyEvents(e) {
    e.addEventListener("keydown", this._onKeyDown), this._domElementKeyEvents = e;
  }
  /**
   * Removes the key event listener previously defined with `listenToKeyEvents()`.
   */
  stopListenToKeyEvents() {
    this._domElementKeyEvents !== null && (this._domElementKeyEvents.removeEventListener("keydown", this._onKeyDown), this._domElementKeyEvents = null);
  }
  /**
   * Save the current state of the controls. This can later be recovered with `reset()`.
   */
  saveState() {
    this.target0.copy(this.target), this.position0.copy(this.object.position), this.zoom0 = this.object.zoom;
  }
  /**
   * Reset the controls to their state from either the last time the `saveState()`
   * was called, or the initial state.
   */
  reset() {
    this.target.copy(this.target0), this.object.position.copy(this.position0), this.object.zoom = this.zoom0, this.object.updateProjectionMatrix(), this.dispatchEvent(Dh), this.update(), this.state = at.NONE;
  }
  update(e = null) {
    const t = this.object.position;
    Et.copy(t).sub(this.target), Et.applyQuaternion(this._quat), this._spherical.setFromVector3(Et), this.autoRotate && this.state === at.NONE && this._rotateLeft(this._getAutoRotationAngle(e)), this.enableDamping ? (this._spherical.theta += this._sphericalDelta.theta * this.dampingFactor, this._spherical.phi += this._sphericalDelta.phi * this.dampingFactor) : (this._spherical.theta += this._sphericalDelta.theta, this._spherical.phi += this._sphericalDelta.phi);
    let i = this.minAzimuthAngle, s = this.maxAzimuthAngle;
    isFinite(i) && isFinite(s) && (i < -Math.PI ? i += qt : i > Math.PI && (i -= qt), s < -Math.PI ? s += qt : s > Math.PI && (s -= qt), i <= s ? this._spherical.theta = Math.max(i, Math.min(s, this._spherical.theta)) : this._spherical.theta = this._spherical.theta > (i + s) / 2 ? Math.max(i, this._spherical.theta) : Math.min(s, this._spherical.theta)), this._spherical.phi = Math.max(this.minPolarAngle, Math.min(this.maxPolarAngle, this._spherical.phi)), this._spherical.makeSafe(), this.enableDamping === !0 ? this.target.addScaledVector(this._panOffset, this.dampingFactor) : this.target.add(this._panOffset), this.target.sub(this.cursor), this.target.clampLength(this.minTargetRadius, this.maxTargetRadius), this.target.add(this.cursor);
    let r = !1;
    if (this.zoomToCursor && this._performCursorZoom || this.object.isOrthographicCamera)
      this._spherical.radius = this._clampDistance(this._spherical.radius);
    else {
      const o = this._spherical.radius;
      this._spherical.radius = this._clampDistance(this._spherical.radius * this._scale), r = o != this._spherical.radius;
    }
    if (Et.setFromSpherical(this._spherical), Et.applyQuaternion(this._quatInverse), t.copy(this.target).add(Et), this.object.lookAt(this.target), this.enableDamping === !0 ? (this._sphericalDelta.theta *= 1 - this.dampingFactor, this._sphericalDelta.phi *= 1 - this.dampingFactor, this._panOffset.multiplyScalar(1 - this.dampingFactor)) : (this._sphericalDelta.set(0, 0, 0), this._panOffset.set(0, 0, 0)), this.zoomToCursor && this._performCursorZoom) {
      let o = null;
      if (this.object.isPerspectiveCamera) {
        const a = Et.length();
        o = this._clampDistance(a * this._scale);
        const l = a - o;
        this.object.position.addScaledVector(this._dollyDirection, l), this.object.updateMatrixWorld(), r = !!l;
      } else if (this.object.isOrthographicCamera) {
        const a = new N(this._mouse.x, this._mouse.y, 0);
        a.unproject(this.object);
        const l = this.object.zoom;
        this.object.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.object.zoom / this._scale)), this.object.updateProjectionMatrix(), r = l !== this.object.zoom;
        const c = new N(this._mouse.x, this._mouse.y, 0);
        c.unproject(this.object), this.object.position.sub(c).add(a), this.object.updateMatrixWorld(), o = Et.length();
      } else
        console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."), this.zoomToCursor = !1;
      o !== null && (this.screenSpacePanning ? this.target.set(0, 0, -1).transformDirection(this.object.matrix).multiplyScalar(o).add(this.object.position) : (_o.origin.copy(this.object.position), _o.direction.set(0, 0, -1).transformDirection(this.object.matrix), Math.abs(this.object.up.dot(_o.direction)) < ty ? this.object.lookAt(this.target) : (Lh.setFromNormalAndCoplanarPoint(this.object.up, this.target), _o.intersectPlane(Lh, this.target))));
    } else if (this.object.isOrthographicCamera) {
      const o = this.object.zoom;
      this.object.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.object.zoom / this._scale)), o !== this.object.zoom && (this.object.updateProjectionMatrix(), r = !0);
    }
    return this._scale = 1, this._performCursorZoom = !1, r || this._lastPosition.distanceToSquared(this.object.position) > Qa || 8 * (1 - this._lastQuaternion.dot(this.object.quaternion)) > Qa || this._lastTargetPosition.distanceToSquared(this.target) > Qa ? (this.dispatchEvent(Dh), this._lastPosition.copy(this.object.position), this._lastQuaternion.copy(this.object.quaternion), this._lastTargetPosition.copy(this.target), !0) : !1;
  }
  _getAutoRotationAngle(e) {
    return e !== null ? qt / 60 * this.autoRotateSpeed * e : qt / 60 / 60 * this.autoRotateSpeed;
  }
  _getZoomScale(e) {
    const t = Math.abs(e * 0.01);
    return Math.pow(0.95, this.zoomSpeed * t);
  }
  _rotateLeft(e) {
    this._sphericalDelta.theta -= e;
  }
  _rotateUp(e) {
    this._sphericalDelta.phi -= e;
  }
  _panLeft(e, t) {
    Et.setFromMatrixColumn(t, 0), Et.multiplyScalar(-e), this._panOffset.add(Et);
  }
  _panUp(e, t) {
    this.screenSpacePanning === !0 ? Et.setFromMatrixColumn(t, 1) : (Et.setFromMatrixColumn(t, 0), Et.crossVectors(this.object.up, Et)), Et.multiplyScalar(e), this._panOffset.add(Et);
  }
  // deltaX and deltaY are in pixels; right and down are positive
  _pan(e, t) {
    const i = this.domElement;
    if (this.object.isPerspectiveCamera) {
      const s = this.object.position;
      Et.copy(s).sub(this.target);
      let r = Et.length();
      r *= Math.tan(this.object.fov / 2 * Math.PI / 180), this._panLeft(2 * e * r / i.clientHeight, this.object.matrix), this._panUp(2 * t * r / i.clientHeight, this.object.matrix);
    } else this.object.isOrthographicCamera ? (this._panLeft(e * (this.object.right - this.object.left) / this.object.zoom / i.clientWidth, this.object.matrix), this._panUp(t * (this.object.top - this.object.bottom) / this.object.zoom / i.clientHeight, this.object.matrix)) : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."), this.enablePan = !1);
  }
  _dollyOut(e) {
    this.object.isPerspectiveCamera || this.object.isOrthographicCamera ? this._scale /= e : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."), this.enableZoom = !1);
  }
  _dollyIn(e) {
    this.object.isPerspectiveCamera || this.object.isOrthographicCamera ? this._scale *= e : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."), this.enableZoom = !1);
  }
  _updateZoomParameters(e, t) {
    if (!this.zoomToCursor)
      return;
    this._performCursorZoom = !0;
    const i = this.domElement.getBoundingClientRect(), s = e - i.left, r = t - i.top, o = i.width, a = i.height;
    this._mouse.x = s / o * 2 - 1, this._mouse.y = -(r / a) * 2 + 1, this._dollyDirection.set(this._mouse.x, this._mouse.y, 1).unproject(this.object).sub(this.object.position).normalize();
  }
  _clampDistance(e) {
    return Math.max(this.minDistance, Math.min(this.maxDistance, e));
  }
  //
  // event callbacks - update the object state
  //
  _handleMouseDownRotate(e) {
    this._rotateStart.set(e.clientX, e.clientY);
  }
  _handleMouseDownDolly(e) {
    this._updateZoomParameters(e.clientX, e.clientX), this._dollyStart.set(e.clientX, e.clientY);
  }
  _handleMouseDownPan(e) {
    this._panStart.set(e.clientX, e.clientY);
  }
  _handleMouseMoveRotate(e) {
    this._rotateEnd.set(e.clientX, e.clientY), this._rotateDelta.subVectors(this._rotateEnd, this._rotateStart).multiplyScalar(this.rotateSpeed);
    const t = this.domElement;
    this._rotateLeft(qt * this._rotateDelta.x / t.clientHeight), this._rotateUp(qt * this._rotateDelta.y / t.clientHeight), this._rotateStart.copy(this._rotateEnd), this.update();
  }
  _handleMouseMoveDolly(e) {
    this._dollyEnd.set(e.clientX, e.clientY), this._dollyDelta.subVectors(this._dollyEnd, this._dollyStart), this._dollyDelta.y > 0 ? this._dollyOut(this._getZoomScale(this._dollyDelta.y)) : this._dollyDelta.y < 0 && this._dollyIn(this._getZoomScale(this._dollyDelta.y)), this._dollyStart.copy(this._dollyEnd), this.update();
  }
  _handleMouseMovePan(e) {
    this._panEnd.set(e.clientX, e.clientY), this._panDelta.subVectors(this._panEnd, this._panStart).multiplyScalar(this.panSpeed), this._pan(this._panDelta.x, this._panDelta.y), this._panStart.copy(this._panEnd), this.update();
  }
  _handleMouseWheel(e) {
    this._updateZoomParameters(e.clientX, e.clientY), e.deltaY < 0 ? this._dollyIn(this._getZoomScale(e.deltaY)) : e.deltaY > 0 && this._dollyOut(this._getZoomScale(e.deltaY)), this.update();
  }
  _handleKeyDown(e) {
    let t = !1;
    switch (e.code) {
      case this.keys.UP:
        e.ctrlKey || e.metaKey || e.shiftKey ? this.enableRotate && this._rotateUp(qt * this.keyRotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(0, this.keyPanSpeed), t = !0;
        break;
      case this.keys.BOTTOM:
        e.ctrlKey || e.metaKey || e.shiftKey ? this.enableRotate && this._rotateUp(-qt * this.keyRotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(0, -this.keyPanSpeed), t = !0;
        break;
      case this.keys.LEFT:
        e.ctrlKey || e.metaKey || e.shiftKey ? this.enableRotate && this._rotateLeft(qt * this.keyRotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(this.keyPanSpeed, 0), t = !0;
        break;
      case this.keys.RIGHT:
        e.ctrlKey || e.metaKey || e.shiftKey ? this.enableRotate && this._rotateLeft(-qt * this.keyRotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(-this.keyPanSpeed, 0), t = !0;
        break;
    }
    t && (e.preventDefault(), this.update());
  }
  _handleTouchStartRotate(e) {
    if (this._pointers.length === 1)
      this._rotateStart.set(e.pageX, e.pageY);
    else {
      const t = this._getSecondPointerPosition(e), i = 0.5 * (e.pageX + t.x), s = 0.5 * (e.pageY + t.y);
      this._rotateStart.set(i, s);
    }
  }
  _handleTouchStartPan(e) {
    if (this._pointers.length === 1)
      this._panStart.set(e.pageX, e.pageY);
    else {
      const t = this._getSecondPointerPosition(e), i = 0.5 * (e.pageX + t.x), s = 0.5 * (e.pageY + t.y);
      this._panStart.set(i, s);
    }
  }
  _handleTouchStartDolly(e) {
    const t = this._getSecondPointerPosition(e), i = e.pageX - t.x, s = e.pageY - t.y, r = Math.sqrt(i * i + s * s);
    this._dollyStart.set(0, r);
  }
  _handleTouchStartDollyPan(e) {
    this.enableZoom && this._handleTouchStartDolly(e), this.enablePan && this._handleTouchStartPan(e);
  }
  _handleTouchStartDollyRotate(e) {
    this.enableZoom && this._handleTouchStartDolly(e), this.enableRotate && this._handleTouchStartRotate(e);
  }
  _handleTouchMoveRotate(e) {
    if (this._pointers.length == 1)
      this._rotateEnd.set(e.pageX, e.pageY);
    else {
      const i = this._getSecondPointerPosition(e), s = 0.5 * (e.pageX + i.x), r = 0.5 * (e.pageY + i.y);
      this._rotateEnd.set(s, r);
    }
    this._rotateDelta.subVectors(this._rotateEnd, this._rotateStart).multiplyScalar(this.rotateSpeed);
    const t = this.domElement;
    this._rotateLeft(qt * this._rotateDelta.x / t.clientHeight), this._rotateUp(qt * this._rotateDelta.y / t.clientHeight), this._rotateStart.copy(this._rotateEnd);
  }
  _handleTouchMovePan(e) {
    if (this._pointers.length === 1)
      this._panEnd.set(e.pageX, e.pageY);
    else {
      const t = this._getSecondPointerPosition(e), i = 0.5 * (e.pageX + t.x), s = 0.5 * (e.pageY + t.y);
      this._panEnd.set(i, s);
    }
    this._panDelta.subVectors(this._panEnd, this._panStart).multiplyScalar(this.panSpeed), this._pan(this._panDelta.x, this._panDelta.y), this._panStart.copy(this._panEnd);
  }
  _handleTouchMoveDolly(e) {
    const t = this._getSecondPointerPosition(e), i = e.pageX - t.x, s = e.pageY - t.y, r = Math.sqrt(i * i + s * s);
    this._dollyEnd.set(0, r), this._dollyDelta.set(0, Math.pow(this._dollyEnd.y / this._dollyStart.y, this.zoomSpeed)), this._dollyOut(this._dollyDelta.y), this._dollyStart.copy(this._dollyEnd);
    const o = (e.pageX + t.x) * 0.5, a = (e.pageY + t.y) * 0.5;
    this._updateZoomParameters(o, a);
  }
  _handleTouchMoveDollyPan(e) {
    this.enableZoom && this._handleTouchMoveDolly(e), this.enablePan && this._handleTouchMovePan(e);
  }
  _handleTouchMoveDollyRotate(e) {
    this.enableZoom && this._handleTouchMoveDolly(e), this.enableRotate && this._handleTouchMoveRotate(e);
  }
  // pointers
  _addPointer(e) {
    this._pointers.push(e.pointerId);
  }
  _removePointer(e) {
    delete this._pointerPositions[e.pointerId];
    for (let t = 0; t < this._pointers.length; t++)
      if (this._pointers[t] == e.pointerId) {
        this._pointers.splice(t, 1);
        return;
      }
  }
  _isTrackingPointer(e) {
    for (let t = 0; t < this._pointers.length; t++)
      if (this._pointers[t] == e.pointerId) return !0;
    return !1;
  }
  _trackPointer(e) {
    let t = this._pointerPositions[e.pointerId];
    t === void 0 && (t = new Ve(), this._pointerPositions[e.pointerId] = t), t.set(e.pageX, e.pageY);
  }
  _getSecondPointerPosition(e) {
    const t = e.pointerId === this._pointers[0] ? this._pointers[1] : this._pointers[0];
    return this._pointerPositions[t];
  }
  //
  _customWheelEvent(e) {
    const t = e.deltaMode, i = {
      clientX: e.clientX,
      clientY: e.clientY,
      deltaY: e.deltaY
    };
    switch (t) {
      case 1:
        i.deltaY *= 16;
        break;
      case 2:
        i.deltaY *= 100;
        break;
    }
    return e.ctrlKey && !this._controlActive && (i.deltaY *= 10), i;
  }
}
function iy(n) {
  this.enabled !== !1 && (this._pointers.length === 0 && (this.domElement.setPointerCapture(n.pointerId), this.domElement.addEventListener("pointermove", this._onPointerMove), this.domElement.addEventListener("pointerup", this._onPointerUp)), !this._isTrackingPointer(n) && (this._addPointer(n), n.pointerType === "touch" ? this._onTouchStart(n) : this._onMouseDown(n)));
}
function sy(n) {
  this.enabled !== !1 && (n.pointerType === "touch" ? this._onTouchMove(n) : this._onMouseMove(n));
}
function ry(n) {
  switch (this._removePointer(n), this._pointers.length) {
    case 0:
      this.domElement.releasePointerCapture(n.pointerId), this.domElement.removeEventListener("pointermove", this._onPointerMove), this.domElement.removeEventListener("pointerup", this._onPointerUp), this.dispatchEvent(Ld), this.state = at.NONE;
      break;
    case 1:
      const e = this._pointers[0], t = this._pointerPositions[e];
      this._onTouchStart({ pointerId: e, pageX: t.x, pageY: t.y });
      break;
  }
}
function oy(n) {
  let e;
  switch (n.button) {
    case 0:
      e = this.mouseButtons.LEFT;
      break;
    case 1:
      e = this.mouseButtons.MIDDLE;
      break;
    case 2:
      e = this.mouseButtons.RIGHT;
      break;
    default:
      e = -1;
  }
  switch (e) {
    case Ps.DOLLY:
      if (this.enableZoom === !1) return;
      this._handleMouseDownDolly(n), this.state = at.DOLLY;
      break;
    case Ps.ROTATE:
      if (n.ctrlKey || n.metaKey || n.shiftKey) {
        if (this.enablePan === !1) return;
        this._handleMouseDownPan(n), this.state = at.PAN;
      } else {
        if (this.enableRotate === !1) return;
        this._handleMouseDownRotate(n), this.state = at.ROTATE;
      }
      break;
    case Ps.PAN:
      if (n.ctrlKey || n.metaKey || n.shiftKey) {
        if (this.enableRotate === !1) return;
        this._handleMouseDownRotate(n), this.state = at.ROTATE;
      } else {
        if (this.enablePan === !1) return;
        this._handleMouseDownPan(n), this.state = at.PAN;
      }
      break;
    default:
      this.state = at.NONE;
  }
  this.state !== at.NONE && this.dispatchEvent(Ic);
}
function ay(n) {
  switch (this.state) {
    case at.ROTATE:
      if (this.enableRotate === !1) return;
      this._handleMouseMoveRotate(n);
      break;
    case at.DOLLY:
      if (this.enableZoom === !1) return;
      this._handleMouseMoveDolly(n);
      break;
    case at.PAN:
      if (this.enablePan === !1) return;
      this._handleMouseMovePan(n);
      break;
  }
}
function ly(n) {
  this.enabled === !1 || this.enableZoom === !1 || this.state !== at.NONE || (n.preventDefault(), this.dispatchEvent(Ic), this._handleMouseWheel(this._customWheelEvent(n)), this.dispatchEvent(Ld));
}
function cy(n) {
  this.enabled !== !1 && this._handleKeyDown(n);
}
function uy(n) {
  switch (this._trackPointer(n), this._pointers.length) {
    case 1:
      switch (this.touches.ONE) {
        case Ss.ROTATE:
          if (this.enableRotate === !1) return;
          this._handleTouchStartRotate(n), this.state = at.TOUCH_ROTATE;
          break;
        case Ss.PAN:
          if (this.enablePan === !1) return;
          this._handleTouchStartPan(n), this.state = at.TOUCH_PAN;
          break;
        default:
          this.state = at.NONE;
      }
      break;
    case 2:
      switch (this.touches.TWO) {
        case Ss.DOLLY_PAN:
          if (this.enableZoom === !1 && this.enablePan === !1) return;
          this._handleTouchStartDollyPan(n), this.state = at.TOUCH_DOLLY_PAN;
          break;
        case Ss.DOLLY_ROTATE:
          if (this.enableZoom === !1 && this.enableRotate === !1) return;
          this._handleTouchStartDollyRotate(n), this.state = at.TOUCH_DOLLY_ROTATE;
          break;
        default:
          this.state = at.NONE;
      }
      break;
    default:
      this.state = at.NONE;
  }
  this.state !== at.NONE && this.dispatchEvent(Ic);
}
function hy(n) {
  switch (this._trackPointer(n), this.state) {
    case at.TOUCH_ROTATE:
      if (this.enableRotate === !1) return;
      this._handleTouchMoveRotate(n), this.update();
      break;
    case at.TOUCH_PAN:
      if (this.enablePan === !1) return;
      this._handleTouchMovePan(n), this.update();
      break;
    case at.TOUCH_DOLLY_PAN:
      if (this.enableZoom === !1 && this.enablePan === !1) return;
      this._handleTouchMoveDollyPan(n), this.update();
      break;
    case at.TOUCH_DOLLY_ROTATE:
      if (this.enableZoom === !1 && this.enableRotate === !1) return;
      this._handleTouchMoveDollyRotate(n), this.update();
      break;
    default:
      this.state = at.NONE;
  }
}
function fy(n) {
  this.enabled !== !1 && n.preventDefault();
}
function dy(n) {
  n.key === "Control" && (this._controlActive = !0, this.domElement.getRootNode().addEventListener("keyup", this._interceptControlUp, { passive: !0, capture: !0 }));
}
function py(n) {
  n.key === "Control" && (this._controlActive = !1, this.domElement.getRootNode().removeEventListener("keyup", this._interceptControlUp, { passive: !0, capture: !0 }));
}
const nr = Object.freeze({
  dry: [3.8, 4.8, 2.25],
  small_room: [2.35, 3.1, 1.5],
  empty_club: [4.82, 6.2, 2.48],
  medium_room: [6.2, 9.3, 2.9],
  cathedral: [8.2, 12.4, 6.8],
  dual_delay: [5.8, 9.6, 3.2],
  outside: [7.2, 11.5, 0]
}), Ih = Object.freeze({
  standard: Object.freeze({
    fov: 38,
    position: Object.freeze([7.5, 6.4, 8.5]),
    target: Object.freeze([0, 1.1, 0]),
    minDistance: 4.2,
    maxDistance: 18
  }),
  compact: Object.freeze({
    fov: 44,
    position: Object.freeze([10.1, 8.13, 11.42]),
    target: Object.freeze([0, 1.35, 0]),
    minDistance: 4.8,
    maxDistance: 22
  })
});
function my(n = !1) {
  return n ? Ih.compact : Ih.standard;
}
function _y(n) {
  return n.space_mode === "dry" ? nr.dry : n.space_mode === "outside" ? nr.outside : n.space_mode === "sfx" ? nr.dual_delay : nr[n.room_preset] ?? nr.medium_room;
}
class gy {
  constructor(e, t = {}) {
    this.canvas = e, this.compact = !!t.compact, this.renderer = new ey({
      canvas: e,
      antialias: !0,
      alpha: !1,
      preserveDrawingBuffer: !0
    }), this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)), this.renderer.outputColorSpace = tn, this.renderer.toneMapping = ed, this.renderer.toneMappingExposure = 1.05, this.scene = new t0();
    const i = my(this.compact);
    this.camera = new nn(i.fov, 1, 0.1, 100), this.camera.position.set(...i.position), this.controls = new ny(this.camera, e), this.controls.enableDamping = !0, this.controls.dampingFactor = 0.06, this.controls.minDistance = i.minDistance, this.controls.maxDistance = i.maxDistance, this.controls.target.set(...i.target), this.root = new Dn(), this.room = new Dn(), this.waveGroup = new Dn(), this.atmosphere = new Dn(), this.root.add(this.room, this.waveGroup, this.atmosphere), this.scene.add(this.root), this.source = this.createSource(), this.listener = this.createListener(), this.root.add(this.source, this.listener), this.hemisphere = new u0(14351338, 1515805, 1.7), this.key = new p0(16777215, 2.6), this.key.position.set(4, 7, 4), this.rim = new f0(7730613, 12, 24), this.rim.position.set(-4, 3, -3), this.scene.add(this.hemisphere, this.key, this.rim), this.clock = new _0(), this.running = !0, this.resizeObserver = new ResizeObserver(() => this.resize()), this.resizeObserver.observe(e.parentElement ?? e), this.resize(), this.animate();
  }
  createSource() {
    const e = new Dn(), t = new vt(
      new Pc(0.2, 3),
      new bo({
        color: 10354640,
        emissive: 3663245,
        emissiveIntensity: 2.4,
        roughness: 0.2
      })
    ), i = new vt(
      new ys(0.34, 24, 24),
      new Rn({
        color: 7536566,
        transparent: !0,
        opacity: 0.09,
        depthWrite: !1
      })
    );
    return e.add(t, i), e;
  }
  createListener() {
    const e = new Dn(), t = new bo({
      color: 16764812,
      emissive: 16747586,
      emissiveIntensity: 1.2
    }), i = new vt(new ys(0.12, 20, 20), t);
    i.position.y = 0.34;
    const s = new vt(new Rc(0.11, 0.26, 6, 12), t), r = new vt(
      new Es(0.34, 0.018, 10, 48),
      new Rn({ color: 16759411, transparent: !0, opacity: 0.55 })
    );
    return r.rotation.x = Math.PI / 2, r.position.y = 0.02, e.add(i, s, r), e;
  }
  clearGroup(e) {
    for (; e.children.length; )
      e.children.pop().traverse?.((i) => {
        i.geometry?.dispose?.(), Array.isArray(i.material) ? i.material.forEach((s) => s.dispose?.()) : i.material?.dispose?.();
      });
  }
  update(e, t, i) {
    this.state = e, this.metrics = t;
    const [s, r, o] = _y(e);
    this.displayDimensions = { width: s, length: r, height: o };
    const [a, l] = i.palette, c = new We(a), u = new We(l);
    this.clearGroup(this.room), this.clearGroup(this.atmosphere), i.mode === "outside" ? this.buildOutside(s, r, c, u, i.time_of_day) : i.mode === "sfx" ? this.buildDualDelay(s, r, o, c, u) : this.buildRoom(s, r, o, c, u, i.mode === "dry"), this.source.position.set(-s * 0.22, 0.34, -r * 0.16), this.listener.position.set(s * 0.16, 0.2, r * 0.22), this.source.children[0].material.color.set(c), this.source.children[0].material.emissive.set(c), this.source.children[1].material.color.set(c), this.buildWaves(c, s, r, t.visual_amount), this.updateLighting(i, c, u);
  }
  buildRoom(e, t, i, s, r, o) {
    const a = new vt(
      new zs(e, t, 12, 12),
      new bo({
        color: r,
        roughness: 0.78,
        metalness: 0.08,
        transparent: !0,
        opacity: o ? 0.34 : 0.82
      })
    );
    a.rotation.x = -Math.PI / 2, this.room.add(a);
    const l = new rh(
      Math.max(e, t),
      16,
      s,
      r.clone().offsetHSL(0, 0, 0.13)
    );
    l.material.transparent = !0, l.material.opacity = o ? 0.06 : 0.16, l.position.y = 6e-3, this.room.add(l);
    const c = new ji(e, i, t), u = new vt(
      c,
      new eh({
        color: s,
        transparent: !0,
        opacity: o ? 0.025 : 0.085,
        side: Wt,
        roughness: 0.24,
        transmission: o ? 0 : 0.18,
        depthWrite: !1
      })
    );
    u.position.y = i / 2;
    const h = new Sd(
      new a0(c),
      new wc({
        color: s,
        transparent: !0,
        opacity: o ? 0.14 : 0.72
      })
    );
    h.position.y = i / 2, this.room.add(u, h);
  }
  buildOutside(e, t, i, s, r) {
    const o = new vt(
      new zs(e, t, 20, 20),
      new eh({
        color: s,
        transparent: !0,
        opacity: 0.72,
        roughness: 0.78,
        metalness: 0.04
      })
    );
    o.rotation.x = -Math.PI / 2, this.room.add(o);
    const a = new rh(Math.max(e, t), 20, i, s);
    a.material.transparent = !0, a.material.opacity = r === "night" ? 0.15 : 0.1, a.position.y = 6e-3, this.room.add(a);
    const l = new vt(
      new Es(Math.max(e, t) * 0.43, 0.012, 8, 96),
      new Rn({ color: i, transparent: !0, opacity: 0.42 })
    );
    l.rotation.x = Math.PI / 2, l.position.y = 0.04, this.room.add(l);
    const c = r === "night" ? 74 : 28, u = new Float32Array(c * 3);
    for (let b = 0; b < c; b += 1)
      u[b * 3] = (Math.random() - 0.5) * e * 1.8, u[b * 3 + 1] = 0.8 + Math.random() * 4.6, u[b * 3 + 2] = (Math.random() - 0.5) * t * 1.2;
    const h = new Nt();
    h.setAttribute("position", new En(u, 3));
    const f = new o0(
      h,
      new yd({
        color: r === "night" ? 13095423 : i,
        size: r === "night" ? 0.035 : 0.022,
        transparent: !0,
        opacity: r === "night" ? 0.78 : 0.32
      })
    );
    this.atmosphere.add(f);
    const p = r === "night", v = p ? 0.22 : 0.5, x = p ? 12174847 : 16760114, m = new vt(
      new ys(v, 24, 24),
      new Rn({
        color: x,
        transparent: !0,
        opacity: p ? 0.88 : 1,
        toneMapped: !1
      })
    ), d = new vt(
      new ys(v * (p ? 1.65 : 2.25), 24, 24),
      new Rn({
        color: x,
        transparent: !0,
        opacity: p ? 0.08 : 0.13,
        depthWrite: !1,
        toneMapped: !1
      })
    );
    m.add(d), m.position.set(-e * 0.38, 3.8, -t * 0.32), this.atmosphere.add(m);
  }
  buildDualDelay(e, t, i, s, r) {
    this.buildRoom(e, t, i, s, r, !1);
    const o = new Rn({
      color: s,
      transparent: !0,
      opacity: 0.64,
      depthWrite: !1
    }), a = new Rn({
      color: s,
      transparent: !0,
      opacity: 0.28,
      depthWrite: !1
    });
    [-1, 1].forEach((l, c) => {
      const u = new vt(
        new ji(0.035, 0.035, t * 0.72),
        o.clone()
      );
      u.position.set(l * e * 0.2, 0.16 + c * 0.08, 0), this.room.add(u);
      for (let h = 0; h < 5; h += 1) {
        const f = new vt(
          new Es(0.26 + h * 0.09, 0.012, 8, 48),
          a.clone()
        );
        f.rotation.x = Math.PI / 2, f.position.set(
          l * e * 0.2,
          0.18 + c * 0.08,
          -t * 0.25 + h * t * 0.125
        ), this.room.add(f);
      }
    });
  }
  buildWaves(e, t, i, s) {
    if (this.clearGroup(this.waveGroup), s <= 0) return;
    const r = 3 + Math.round(s * 4);
    for (let o = 0; o < r; o += 1) {
      const a = new vt(
        new Es(0.35, 0.012, 8, 64),
        new Rn({ color: e, transparent: !0, opacity: 0.18, depthWrite: !1 })
      );
      a.rotation.x = Math.PI / 2, a.position.copy(this.source.position), a.userData = { index: o, count: r, maxScale: Math.max(t, i) * 1.2 }, this.waveGroup.add(a);
    }
  }
  updateLighting(e, t, i) {
    e.mode === "outside" && e.time_of_day === "day" ? (this.scene.background = new We(2893592), this.hemisphere.color.set(16773053), this.hemisphere.groundColor.set(4011808), this.key.color.set(16762972), this.key.intensity = 4.1) : e.mode === "outside" ? (this.scene.background = new We(329751), this.hemisphere.color.set(9215999), this.hemisphere.groundColor.set(592660), this.key.color.set(8624127), this.key.intensity = 1.3) : (this.scene.background = i.clone().multiplyScalar(0.18), this.hemisphere.color.set(14351338), this.hemisphere.groundColor.set(i), this.key.color.set(16777215), this.key.intensity = e.mode === "dry" ? 1.6 : 2.6), this.rim.color.set(t), this.rim.intensity = e.mode === "dry" ? 4 : 12;
  }
  resize() {
    const e = this.canvas.parentElement, t = Math.max(1, e?.clientWidth ?? this.canvas.clientWidth), i = Math.max(1, e?.clientHeight ?? this.canvas.clientHeight);
    this.renderer.setSize(t, i, !1), this.camera.aspect = t / i, this.camera.updateProjectionMatrix();
  }
  animate = () => {
    if (!this.running) return;
    const e = this.clock.getElapsedTime(), t = this.metrics?.visual_amount ?? 0, i = 0.22 + t * 0.22;
    for (const s of this.waveGroup.children) {
      const r = (e * i + s.userData.index / s.userData.count) % 1, o = 0.2 + r * s.userData.maxScale;
      s.scale.setScalar(o), s.material.opacity = (1 - r) * (0.08 + t * 0.22);
    }
    this.source.rotation.y = e * 0.4, this.atmosphere.rotation.y = Math.sin(e * 0.08) * 0.035, this.controls.update(), this.renderer.render(this.scene, this.camera), this.animationFrame = requestAnimationFrame(this.animate);
  };
  toDataURL() {
    return this.renderer.render(this.scene, this.camera), this.canvas.toDataURL("image/png");
  }
  dispose() {
    this.running = !1, cancelAnimationFrame(this.animationFrame), this.resizeObserver.disconnect(), this.controls.dispose(), this.clearGroup(this.root), this.renderer.dispose();
  }
}
const vy = (n, e) => {
  const t = n.__vccOpts || n;
  for (const [i, s] of e)
    t[i] = s;
  return t;
}, xy = { class: "akuspace-widget__header" }, My = ["aria-expanded", "aria-label"], Sy = { class: "akuspace-widget__mode-fader" }, yy = ["max", "value"], Ey = {
  key: 0,
  class: "akuspace-widget__faders"
}, Ty = {
  key: 1,
  class: "akuspace-widget__faders"
}, by = ["max", "value"], Ay = ["max", "value"], wy = {
  key: 2,
  class: "akuspace-widget__faders"
}, Ry = { class: "akuspace-widget__segments akuspace-widget__segments--two" }, Cy = ["onClick"], Py = {
  key: 3,
  class: "akuspace-widget__faders"
}, Dy = ["max", "value"], Ly = { class: "akuspace-widget__ticks akuspace-widget__ticks--two" }, Iy = {
  __name: "AcousticSpaceWidget",
  props: {
    initialState: { type: Object, default: () => ({}) },
    onStateChange: { type: Function, default: null }
  },
  setup(n, { expose: e }) {
    const t = n, i = /* @__PURE__ */ Ws(null), s = /* @__PURE__ */ Ws(null), r = /* @__PURE__ */ Ws(null), o = /* @__PURE__ */ gr({ ...pi, ...t.initialState }), a = /* @__PURE__ */ Ws(!1), l = /* @__PURE__ */ Ws(!1), c = /* @__PURE__ */ gr({ x: 0, y: 0 });
    let u = null, h = null, f = null;
    const p = un(() => Zf(o)), v = un(() => H_(o)), x = un(() => Math.max(
      0,
      ts.findIndex((me) => me.value === o.space_mode)
    )), m = un(() => ts[x.value]?.label ?? "Room"), d = un(() => Math.max(0, vs.indexOf(o.room_preset))), b = un(() => Math.max(0, xs.indexOf(o.effect_level))), A = un(() => Math.max(0, Ms.indexOf(o.sfx_level))), M = un(() => o.sfx_level === "high" ? "High" : "Low"), R = un(() => a.value || l.value), w = un(() => ({
      transform: `translate(calc(-50% + ${c.x}px), ${c.y}px)`
    })), D = un(() => o.space_mode === "dry" ? "Application · Off" : o.space_mode === "outside" ? `Space · ${o.outdoor_time === "night" ? "Night" : "Day"}` : o.space_mode === "sfx" ? `Sound effects · ${M.value}` : `Room · ${Mo[o.effect_level]?.label ?? "Moderate"}`);
    function U(me) {
      return { gridTemplateColumns: `repeat(${me}, 1fr)` };
    }
    function y() {
      u?.update(o, v.value, p.value);
    }
    function S(me = {}) {
      Object.assign(o, me);
    }
    function P() {
      h !== null && window.clearTimeout(h), h = null;
    }
    function L() {
      P(), l.value = !0;
    }
    function V() {
      P(), h = window.setTimeout(() => {
        l.value = !1;
      }, 180);
    }
    function Z() {
      P(), a.value = !a.value, l.value = a.value;
    }
    function te() {
      P(), a.value = !1, l.value = !1;
    }
    function $(me, de, Le) {
      return Math.min(Le, Math.max(de, me));
    }
    function ie(me) {
      if (!f || !i.value || !r.value) return;
      const de = i.value.getBoundingClientRect(), Le = r.value.getBoundingClientRect(), tt = Math.max(0, (de.width - Le.width) / 2 - 8), Ze = Math.max(0, de.height - Le.height - 46);
      c.x = $(f.x + me.clientX - f.clientX, -tt, tt), c.y = $(f.y + me.clientY - f.clientY, 0, Ze);
    }
    function H() {
      f = null, window.removeEventListener("pointermove", ie), window.removeEventListener("pointerup", H), window.removeEventListener("pointercancel", H);
    }
    function fe(me) {
      P(), a.value = !0, f = {
        clientX: me.clientX,
        clientY: me.clientY,
        x: c.x,
        y: c.y
      }, window.addEventListener("pointermove", ie), window.addEventListener("pointerup", H, { once: !0 }), window.addEventListener("pointercancel", H, { once: !0 });
    }
    function xe() {
      P(), H(), u?.dispose(), u = null;
    }
    return vo(
      o,
      () => {
        y(), t.onStateChange?.({ ...o });
      },
      { deep: !0 }
    ), pc(() => {
      u = new gy(s.value, { compact: !0 }), y();
    }), mc(xe), e({ setState: S, cleanup: xe }), (me, de) => (Lt(), Ot("div", {
      ref_key: "rootRef",
      ref: i,
      class: "akuspace-widget",
      style: _i({ "--ak-accent": p.value.palette[0] })
    }, [
      Be("canvas", {
        ref_key: "canvasRef",
        ref: s,
        "aria-label": "Interactive AKUSPACE room preview"
      }, null, 512),
      Be("div", xy, [
        de[4] || (de[4] = Be("span", null, "AKUSPACE", -1)),
        Be("button", {
          class: "akuspace-widget__toggle",
          type: "button",
          "aria-expanded": R.value,
          "aria-label": R.value ? "Fold acoustic controls" : "Open acoustic controls",
          onMouseenter: L,
          onMouseleave: V,
          onClick: Z
        }, [
          Be("i", {
            class: pr({ open: R.value })
          }, null, 2)
        ], 40, My),
        Be("strong", null, cn(p.value.label), 1)
      ]),
      Kt(Km, { name: "akuspace-panel" }, {
        default: df(() => [
          Op(Be("div", {
            ref_key: "panelRef",
            ref: r,
            class: "akuspace-widget__controls",
            style: _i(w.value),
            onMouseenter: L,
            onMouseleave: V
          }, [
            Be("button", {
              class: "akuspace-widget__dragbar",
              type: "button",
              "aria-label": "Move acoustic controls",
              onPointerdown: S_(fe, ["prevent"])
            }, [
              de[5] || (de[5] = Be("i", null, null, -1)),
              Be("span", null, cn(D.value), 1),
              de[6] || (de[6] = Be("i", null, null, -1))
            ], 32),
            Be("label", Sy, [
              Be("span", null, [
                de[7] || (de[7] = Be("span", null, "Mode", -1)),
                Be("strong", null, cn(m.value), 1)
              ]),
              Be("input", {
                type: "range",
                min: "0",
                max: yt(ts).length - 1,
                step: "1",
                value: x.value,
                "aria-label": "AKUSPACE mode",
                onInput: de[0] || (de[0] = (Le) => S({ space_mode: yt(ts)[Number(Le.target.value)].value }))
              }, null, 40, yy),
              Be("small", {
                class: "akuspace-widget__ticks",
                style: _i(U(yt(ts).length))
              }, [
                (Lt(!0), Ot(Vt, null, Ys(yt(ts), (Le) => (Lt(), Ot("i", {
                  key: Le.value
                }, cn(Le.value === "sfx" ? "SFX" : Le.label), 1))), 128))
              ], 4)
            ]),
            o.space_mode === "dry" ? (Lt(), Ot("div", Ey, [...de[8] || (de[8] = [
              Be("div", { class: "akuspace-widget__effect" }, [
                Be("span", null, "AKUSPACE"),
                Be("small", null, "Off")
              ], -1)
            ])])) : sr("", !0),
            o.space_mode === "room" ? (Lt(), Ot("div", Ty, [
              Be("label", null, [
                Be("span", null, [
                  de[9] || (de[9] = Be("span", null, "Reverb size", -1)),
                  Be("strong", null, cn(p.value.short_label), 1)
                ]),
                Be("input", {
                  type: "range",
                  min: "0",
                  max: yt(vs).length - 1,
                  step: "1",
                  value: d.value,
                  onInput: de[1] || (de[1] = (Le) => S({ room_preset: yt(vs)[Number(Le.target.value)] }))
                }, null, 40, by),
                Be("small", {
                  class: "akuspace-widget__ticks",
                  style: _i(U(yt(vs).length))
                }, [
                  (Lt(!0), Ot(Vt, null, Ys(yt(vs), (Le) => (Lt(), Ot("i", { key: Le }, cn(yt($f)[Le].short_label), 1))), 128))
                ], 4)
              ]),
              Be("label", null, [
                Be("span", null, [
                  de[10] || (de[10] = Be("span", null, "Dry / wet", -1)),
                  Be("strong", null, cn(yt(Mo)[o.effect_level]?.label), 1)
                ]),
                Be("input", {
                  type: "range",
                  min: "0",
                  max: yt(xs).length - 1,
                  step: "1",
                  value: b.value,
                  onInput: de[2] || (de[2] = (Le) => S({ effect_level: yt(xs)[Number(Le.target.value)] }))
                }, null, 40, Ay),
                Be("small", {
                  class: "akuspace-widget__ticks",
                  style: _i(U(yt(xs).length))
                }, [
                  (Lt(!0), Ot(Vt, null, Ys(yt(xs), (Le) => (Lt(), Ot("i", { key: Le }, cn(yt(Mo)[Le].label), 1))), 128))
                ], 4)
              ])
            ])) : sr("", !0),
            o.space_mode === "outside" ? (Lt(), Ot("div", wy, [
              Be("div", Ry, [
                (Lt(!0), Ot(Vt, null, Ys(yt(N_), (Le) => (Lt(), Ot("button", {
                  key: Le,
                  class: pr({ active: o.outdoor_time === Le }),
                  onClick: (tt) => S({ outdoor_time: Le })
                }, cn(Le === "day" ? "Day" : "Night"), 11, Cy))), 128))
              ])
            ])) : sr("", !0),
            o.space_mode === "sfx" ? (Lt(), Ot("div", Py, [
              de[12] || (de[12] = Be("div", { class: "akuspace-widget__effect" }, [
                Be("span", null, "Dual Delay"),
                Be("small", null, "Experimental SFX")
              ], -1)),
              Be("label", null, [
                Be("span", null, [
                  de[11] || (de[11] = Be("span", null, "Dry / wet", -1)),
                  Be("strong", null, cn(M.value), 1)
                ]),
                Be("input", {
                  type: "range",
                  min: "0",
                  max: yt(Ms).length - 1,
                  step: "1",
                  value: A.value,
                  onInput: de[3] || (de[3] = (Le) => S({ sfx_level: yt(Ms)[Number(Le.target.value)] }))
                }, null, 40, Dy),
                Be("small", Ly, [
                  (Lt(!0), Ot(Vt, null, Ys(yt(Ms), (Le) => (Lt(), Ot("i", { key: Le }, cn(Le === "high" ? "High" : "Low"), 1))), 128))
                ])
              ])
            ])) : sr("", !0),
            Be("button", {
              class: "akuspace-widget__fold",
              type: "button",
              onClick: te
            }, [...de[13] || (de[13] = [
              Be("i", null, null, -1),
              Be("span", null, "Fold controls", -1),
              Be("i", null, null, -1)
            ])])
          ], 36), [
            [t_, R.value]
          ])
        ]),
        _: 1
      })
    ], 4));
  }
}, Uy = /* @__PURE__ */ vy(Iy, [["__scopeId", "data-v-f88d73ed"]]), { app: Uc } = window.comfyAPI.app;
(() => {
  const n = "akuspace-spatial-widget-styles";
  if (document.getElementById(n)) return;
  const e = document.createElement("link");
  e.id = n, e.rel = "stylesheet", e.href = new URL(
    /* @vite-ignore */
    "./assets/acoustic-space-widget.css",
    import.meta.url
  ).href, document.head.appendChild(e);
})();
const bs = /* @__PURE__ */ new WeakMap(), Nc = "akuspaceSpatialState", Ny = 200, Id = [
  "space_mode",
  "application",
  "room_preset",
  "effect_level",
  "outdoor_time",
  "sfx_level"
], Fy = {
  Off: "dry",
  Room: "room",
  Space: "outside",
  "Sound effects": "sfx",
  dry: "dry",
  room: "room",
  outside: "outside",
  sfx: "sfx"
}, Oy = {
  dry: "Off",
  room: "Room",
  outside: "Space",
  sfx: "Sound effects"
}, Uh = {
  dry: ["Off"],
  room: ["Low", "Moderate", "Heavy"],
  outside: ["Day", "Night"],
  sfx: ["Low", "High"]
}, By = /* @__PURE__ */ new Set([
  "room_preset",
  "effect_level",
  "outdoor_time",
  "sfx_level"
]), zy = /* @__PURE__ */ new Set([
  "AcousticSpaceReference",
  "AcousticSpaceTextEncode",
  "Koshi_AKUSPACEPrompt",
  "Koshi_AKUSPACETextEncode"
]), Hy = /* @__PURE__ */ new Set([
  "AcousticSpaceTextEncode",
  "Koshi_AKUSPACETextEncode"
]);
function Ud(n) {
  return Fy[n] ?? pi.space_mode;
}
function Nd(n) {
  return n.space_mode === "dry" ? "Off" : n.space_mode === "outside" ? n.outdoor_time === "night" ? "Night" : "Day" : n.space_mode === "sfx" ? n.sfx_level === "high" ? "High" : "Low" : n.effect_level === "low" ? "Low" : n.effect_level === "high" ? "Heavy" : "Moderate";
}
function Fd(n, e) {
  return n.space_mode === "outside" && (n.outdoor_time = e === "Night" ? "night" : "day"), n.space_mode === "sfx" && (n.sfx_level = e === "High" ? "high" : "low"), n.space_mode === "room" && (n.effect_level = e === "Low" ? "low" : e === "Heavy" ? "high" : "mid"), n;
}
function _s(n, e, t) {
  return n.widgets?.find((i) => i.name === e)?.value ?? t;
}
function Nh(n) {
  const e = n.properties?.[Nc];
  return e && typeof e == "object" ? e : null;
}
function Fc(n) {
  const e = Nh(n) ?? {}, t = {
    space_mode: Ud(
      e.space_mode ?? _s(n, "space_mode", pi.space_mode)
    ),
    room_preset: e.room_preset ?? _s(n, "room_preset", pi.room_preset),
    effect_level: e.effect_level ?? _s(n, "effect_level", pi.effect_level),
    outdoor_time: e.outdoor_time ?? _s(n, "outdoor_time", pi.outdoor_time),
    sfx_preset: pi.sfx_preset,
    sfx_level: e.sfx_level ?? _s(n, "sfx_level", pi.sfx_level)
  };
  return Nh(n) || Fd(t, _s(n, "application", Nd(t))), t;
}
function Od(n, e) {
  n.properties || (n.properties = {}), n.properties[Nc] = { ...e };
}
function Oc(n, e) {
  for (const t of Id) {
    const i = n.widgets?.find((s) => s.name === t);
    i && (t === "space_mode" ? i.value = Oy[e.space_mode] ?? "Room" : t === "application" ? (i.options.values = Uh[e.space_mode] ?? Uh.room, i.value = Nd(e)) : e[t] !== void 0 && (i.value = e[t]));
  }
}
function Vy(n) {
  for (const e of n.widgets ?? [])
    !By.has(e.name) || e.__akuspaceHidden || (e.__akuspaceHidden = !0, e.__akuspaceOriginalType = e.type, e.__akuspaceOriginalComputeSize = e.computeSize, e.__akuspaceOriginalDraw = e.draw, e.type = "converted-widget", e.computeSize = () => [0, -4], e.draw = () => {
    }, e.hidden = !0, e.options = { ...e.options, hidden: !0 });
}
function ky(n) {
  const e = n.outputs?.[0];
  if (!e) return;
  const t = zd(n) ? "Conditioning" : "Prompt";
  e.name = t, e.label = t, e.localized_name = t;
}
function el(n) {
  Vy(n), ky(n);
  const [e, t] = n.size, i = zd(n) ? 560 : 470;
  n.setSize([Math.max(e, 360), Math.max(t, i)]), Uc.graph?.setDirtyCanvas(!0, !0);
}
function Gy(n) {
  const e = document.createElement("div");
  e.className = "akuspace-widget-host", e.style.cssText = "width:100%;height:100%;min-height:300px";
  const t = {
    container: e,
    currentNode: n,
    widget: null,
    cleanupTimer: null,
    vueApp: null,
    exposed: null
  }, i = T_(Uy, {
    initialState: Fc(n),
    onStateChange: (s) => {
      const r = t.currentNode;
      Oc(r, s), Od(r, s), Uc.graph?.setDirtyCanvas(!0, !0);
    }
  });
  return t.exposed = i.mount(e), t.vueApp = i, bs.set(n, t), t;
}
function Wy(n, e) {
  for (const t of Id) {
    const i = n.widgets?.find((r) => r.name === t);
    if (!i || i.__akuspaceBound) continue;
    const s = i.callback;
    i.callback = function(...r) {
      s?.apply(this, r);
      const o = Fc(n);
      t === "space_mode" ? o.space_mode = Ud(i.value) : t === "application" ? Fd(o, i.value) : o[t] = i.value, Oc(n, o), Od(n, o), e.exposed.setState(o);
    }, i.__akuspaceBound = !0;
  }
}
function Xy(n) {
  let e = bs.get(n);
  e ? (e.cleanupTimer !== null && clearTimeout(e.cleanupTimer), e.cleanupTimer = null, e.currentNode = n, e.exposed.setState(Fc(n))) : e = Gy(n);
  const t = n.addDOMWidget(
    "space_preview",
    "akuspace-spatial-preview",
    e.container,
    { getMinHeight: () => 300, hideOnZoom: !1, serialize: !1 }
  );
  e.widget = t, Wy(n, e);
  const i = t.onRemove?.bind(t);
  t.onRemove = () => {
    i?.();
    const s = bs.get(n);
    !s || s.widget !== t || (s.cleanupTimer = window.setTimeout(() => {
      const r = bs.get(n);
      !r || r.widget !== t || (r.exposed.cleanup(), r.vueApp.unmount(), bs.delete(n));
    }, Ny));
  };
}
function Yy(n, e) {
  const t = n.onPropertyChanged;
  n.onPropertyChanged = function(i, s) {
    t?.call(this, i, s), !(i !== Nc || !s || typeof s != "object") && (Oc(n, s), e.exposed.setState(s));
  };
}
function Bd(n) {
  return [
    n?.constructor?.comfyClass,
    n?.comfyClass,
    n?.type,
    n?.constructor?.type
  ].find((t) => zy.has(t));
}
function zd(n) {
  return Hy.has(Bd(n));
}
function qy(n) {
  return !!Bd(n);
}
Uc.registerExtension({
  name: "Koshi.AKUSPACE.SpatialControl",
  nodeCreated(n) {
    if (!qy(n)) return;
    el(n), Xy(n);
    const e = bs.get(n);
    e && Yy(n, e), window.requestAnimationFrame(() => el(n)), window.setTimeout(() => el(n), 120);
  }
});
